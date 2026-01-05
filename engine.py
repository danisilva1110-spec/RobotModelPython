import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.octave import octave_code

# ==============================================================================
# 1. ENGINE MATEMÁTICA - STANDARD (AR / SECO)
# ==============================================================================
class RobotMathEngine:
    def __init__(self, joint_config, link_vectors_mask, logger_callback=None):
        self.joint_config = joint_config
        self.link_vectors_mask = [sp.Matrix(v) for v in link_vectors_mask]
        self.log = logger_callback if logger_callback else print

        self.t = sp.symbols('t')
        self.g = sp.symbols('g')
        
        self.q, self.dq, self.params_list = [], [], []
        self.frames, self.rotation_matrices = [], []
        self.com_positions_global, self.angular_velocities = [], []
        self.masses = []

        # --- CORREÇÃO FUNDAMENTAL AQUI ---
        # Adiciona a gravidade na lista de parâmetros para o Lambdify reconhecê-la
        self.params_list.append(self.g) 

        self.M, self.G_vec, self.C_total, self.Jacobian = None, None, None, None

    def _rot_matrix_local(self, axis, angle):
        c, s = sp.cos(angle), sp.sin(angle)
        if axis == 'x': return sp.Matrix([[1,0,0],[0,c,-s],[0,s,c]])
        if axis == 'y': return sp.Matrix([[c,0,s],[0,1,0],[-s,0,c]])
        if axis == 'z': return sp.Matrix([[c,-s,0],[s,c,0],[0,0,1]])
        return sp.eye(3)

    def _get_axis_vector(self, axis):
        if axis == 'x': return sp.Matrix([1, 0, 0])
        if axis == 'y': return sp.Matrix([0, 1, 0])
        if axis == 'z': return sp.Matrix([0, 0, 1])
        return sp.Matrix([0,0,0])

    def run_full_process(self):
        self.step_1_kinematics()
        self.step_2_jacobian_M_G()
        if len(self.q) > 8:
            self.log("⚠️ AVISO: Sistema grande. Coriolis pode demorar...")
        self.step_3_coriolis_combined()
        return self.step_4_prepare_export()

    def step_1_kinematics(self):
        self.log("1. (AR) Calculando Cinemática (Com CM Arbitrário)...")
        T_acc = sp.eye(4)
        R_acc = sp.eye(3)
        omega_acc = sp.Matrix([0, 0, 0])

        for i, (j_type, link_vec) in enumerate(zip(self.joint_config, self.link_vectors_mask)):
            q = dynamicsymbols(f'q{i+1}')
            dq = dynamicsymbols(f'q{i+1}', 1)
            self.q.append(q)
            self.dq.append(dq)
            
            m = sp.symbols(f'm{i+1}')
            L = sp.symbols(f'L{i+1}')
            
            # CM Local (cx, cy, cz)
            cx, cy, cz = sp.symbols(f'cx{i+1} cy{i+1} cz{i+1}')
            
            self.masses.append(m)
            self.params_list.extend([m, L, cx, cy, cz])

            type_char, axis_char = j_type[0], j_type[1].lower()
            axis_vec_local = self._get_axis_vector(axis_char)

            if type_char == 'R':
                axis_vec_global = R_acc * axis_vec_local
                omega_new = omega_acc + axis_vec_global * dq
                R_j = self._rot_matrix_local(axis_char, q)
                P_j = sp.Matrix([0,0,0])
            elif type_char == 'D':
                omega_new = omega_acc
                R_j = sp.eye(3)
                P_j = sp.Matrix([0,0,0])
                if axis_char == 'x': P_j[0] = q
                if axis_char == 'y': P_j[1] = q
                if axis_char == 'z': P_j[2] = q

            R_acc = R_acc * R_j
            self.rotation_matrices.append(R_acc)
            self.angular_velocities.append(omega_new)
            omega_acc = omega_new

            T_joint = sp.eye(4)
            T_joint[0:3, 0:3] = R_j
            T_joint[0:3, 3] = P_j
            T_at_joint_start = T_acc * T_joint
            
            P_link_vec = link_vec * L
            T_link = sp.eye(4)
            T_link[0:3, 3] = P_link_vec
            T_acc = T_at_joint_start * T_link
            self.frames.append(T_acc)
            
            # Cálculo do CM Global
            v_cm_local = sp.Matrix([cx, cy, cz, 1])
            p_cm_global = T_at_joint_start * v_cm_local
            self.com_positions_global.append(p_cm_global[0:3, 0])

    def step_2_jacobian_M_G(self):
        self.log("2. (AR) Dinâmica M e G (Steiner + Tensores)...")
        n = len(self.q)
        self.M = sp.zeros(n, n)
        V_tot = 0

        for i in range(n):
            m = self.masses[i]
            pos_cm = self.com_positions_global[i]
            R_global = self.rotation_matrices[i]
            J_v = pos_cm.jacobian(self.q)
            J_w = self.angular_velocities[i].jacobian(self.dq)

            # Inércias
            Ixx, Iyy, Izz = sp.symbols(f'Ixx{i+1} Iyy{i+1} Izz{i+1}')
            self.params_list.extend([Ixx, Iyy, Izz])
            
            I_local = sp.Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
            I_global = R_global * I_local * R_global.T

            self.M += m * J_v.T * J_v + J_w.T * I_global * J_w
            
            # Energia Potencial (Usa self.g)
            V_tot += m * self.g * pos_cm[2]

        self.G_vec = sp.Matrix([V_tot]).jacobian(self.q).T
        J_lin = self.frames[-1][0:3, 3].jacobian(self.q)
        J_ang = self.angular_velocities[-1].jacobian(self.dq)
        self.Jacobian = J_lin.col_join(J_ang)

    def step_3_coriolis_combined(self):
        self.log("3. Calculando Coriolis...")
        n = len(self.q)
        self.C_total = sp.zeros(n, 1)
        dM_dq = [self.M.diff(qk) for qk in self.q]

        for i in range(n):
            termo_linha = 0
            for j in range(n):
                for k in range(j, n): 
                    dM_ij_dk = dM_dq[k][i, j]
                    dM_ik_dj = dM_dq[j][i, k]
                    dM_jk_di = dM_dq[i][j, k]
                    c_ijk = sp.Rational(1,2) * (dM_ij_dk + dM_ik_dj - dM_jk_di)
                    if c_ijk != 0:
                        termo = c_ijk * self.dq[j] * self.dq[k]
                        if k != j: termo *= 2
                        termo_linha += termo
            self.C_total[i] = sp.collect(termo_linha, self.dq)

    def step_4_prepare_export(self):
        self.log("4. Otimizando equações...")
        mapa_subs = {}
        for i in range(len(self.q)):
            mapa_subs[self.q[i]] = sp.Symbol(f'q{i+1}')
            mapa_subs[self.dq[i]] = sp.Symbol(f'dq{i+1}')

        dq_vec = sp.Matrix(self.dq)
        V_cartesian = self.Jacobian * dq_vec if self.Jacobian is not None else None

        return {
            "M": self.M, "G": self.G_vec, "C": self.C_total, "J": self.Jacobian,
            "FK_Pos": self.frames[-1], "FK_Vel": V_cartesian, "Subs": mapa_subs, "Mode": "Air"
        }

# ==============================================================================
# 2. ENGINE MATEMÁTICA - HYDRO (ÁGUA / UVMS)
# ==============================================================================
class RobotMathHydro(RobotMathEngine):
    def __init__(self, joint_config, link_vectors_mask, logger_callback=None):
        super().__init__(joint_config, link_vectors_mask, logger_callback)
        self.rho = sp.symbols('rho')
        self.volumes = []

    def run_full_process(self):
        self.step_1_kinematics_hydro()
        self.step_2_jacobian_M_G() 
        if len(self.q) > 8:
            self.log("⚠️ AVISO: UVMS Grande. Coriolis hidrodinâmico será pesado.")
        self.step_3_coriolis_combined()
        return self.step_4_prepare_export()
    
    def step_1_kinematics_hydro(self):
        super().step_1_kinematics()
        self.log("   -> Adicionando parâmetros hidrodinâmicos...")
        for i in range(len(self.q)):
            vol = sp.symbols(f'vol{i+1}')
            self.volumes.append(vol)
            self.params_list.append(vol)

    def step_2_jacobian_M_G(self):
        self.log("2. (ÁGUA) Dinâmica: M (Inércia + Added Mass) e G (Peso - Empuxo)...")
        n = len(self.q)
        self.M = sp.zeros(n, n)
        V_tot = 0

        for i in range(n):
            m = self.masses[i]
            vol = self.volumes[i]
            pos_cm = self.com_positions_global[i]
            R_global = self.rotation_matrices[i]

            J_v = pos_cm.jacobian(self.q)
            J_w = self.angular_velocities[i].jacobian(self.dq)

            Ixx, Iyy, Izz = sp.symbols(f'Ixx{i+1} Iyy{i+1} Izz{i+1}')
            self.params_list.extend([Ixx, Iyy, Izz])
            I_local_RB = sp.Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
            I_global_RB = R_global * I_local_RB * R_global.T
            
            ma_u, ma_v, ma_w = sp.symbols(f'ma_u{i+1} ma_v{i+1} ma_w{i+1}')
            ma_p, ma_q, ma_r = sp.symbols(f'ma_p{i+1} ma_q{i+1} ma_r{i+1}')
            self.params_list.extend([ma_u, ma_v, ma_w, ma_p, ma_q, ma_r])

            MA_lin_local = sp.Matrix([[ma_u, 0, 0], [0, ma_v, 0], [0, 0, ma_w]])
            MA_rot_local = sp.Matrix([[ma_p, 0, 0], [0, ma_q, 0], [0, 0, ma_r]])

            MA_lin_global = R_global * MA_lin_local * R_global.T
            MA_rot_global = R_global * MA_rot_local * R_global.T

            self.M += J_v.T * (m * sp.eye(3) + MA_lin_global) * J_v
            self.M += J_w.T * (I_global_RB + MA_rot_global) * J_w

            # Potencial (W - B) - Usa self.g
            peso_aparente = (m - self.rho * vol) * self.g
            V_tot += peso_aparente * pos_cm[2]

        self.G_vec = sp.Matrix([V_tot]).jacobian(self.q).T
        J_lin = self.frames[-1][0:3, 3].jacobian(self.q)
        J_ang = self.angular_velocities[-1].jacobian(self.dq)
        self.Jacobian = J_lin.col_join(J_ang)

    def step_4_prepare_export(self):
        data = super().step_4_prepare_export()
        data["Mode"] = "Hydro"
        return data