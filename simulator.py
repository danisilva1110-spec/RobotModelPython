import numpy as np
import sympy as sp

class RobotSimulator:
    def __init__(self, robot_math_instance, mode="Air"):
        self.bot = robot_math_instance
        self.mode = mode
        self.num_dof = len(self.bot.q)
        self.params_values = {} 

        print(f"[{mode}] Compilando equações (Isso pode demorar um pouco)...")
        
        # 1. Identifica Juntas Rotacionais (para wrap)
        self.is_rotational = []
        for j_type in self.bot.joint_config:
            if j_type.startswith('R'):
                self.is_rotational.append(True)
            else:
                self.is_rotational.append(False)
        self.is_rotational = np.array(self.is_rotational, dtype=bool)
        
        # 2. Variáveis Simbólicas
        self.sym_vars = self.bot.q + self.bot.dq + self.bot.params_list
        if hasattr(self.bot, 'rho'):
            self.sym_vars.append(self.bot.rho)
        
        # 3. Compila Funções Dinâmicas (M, C, G)
        self.func_M = sp.lambdify(self.sym_vars, self.bot.M, modules='numpy')
        self.func_C = sp.lambdify(self.sym_vars, self.bot.C_total, modules='numpy')
        self.func_G = sp.lambdify(self.sym_vars, self.bot.G_vec, modules='numpy')
        
        # 4. Compila Jacobiano e FK
        self.func_J = sp.lambdify(self.sym_vars, self.bot.Jacobian, modules='numpy')

        self.funcs_fk_all_links = []
        for frame in self.bot.frames:
            pos_expr = frame[:3, 3] # Pega X, Y, Z
            f_fk = sp.lambdify(self.sym_vars, pos_expr, modules='numpy')
            self.funcs_fk_all_links.append(f_fk)

        print("Compilação concluída!")

    def set_parameters(self, user_values_dict):
        self.params_values = user_values_dict

    def _build_args(self, q, dq):
        p_vals = [self.params_values[str(p)] for p in self.bot.params_list]
        args = list(q) + list(dq) + p_vals
        if hasattr(self.bot, 'rho'):
            args.append(self.params_values['rho'])
        return args
    
    def _wrap_to_pi(self, error_vector):
        """ Força o erro a ficar entre -PI e +PI (Menor Caminho) """
        wrapped = (error_vector + np.pi) % (2 * np.pi) - np.pi
        return np.where(self.is_rotational, wrapped, error_vector)

    def trajectory_planning(self, t, t_total, Pi, Pf):
        """ Trajetória Polinomial Cúbica """
        if t >= t_total: return Pf, np.zeros(3), np.zeros(3)
        
        tau = t / t_total
        s = 3*(tau**2) - 2*(tau**3) 
        sd = (6*tau - 6*(tau**2)) / t_total
        sdd = (6 - 12*tau) / (t_total**2)

        d = Pf - Pi
        P = Pi + d * s
        V = d * sd
        A = d * sdd
        return P, V, A

    def solve_ik_numerical(self, target_pos, q_curr, dt):
        """ 
        Cinemática Inversa com Damped Least Squares + NULL SPACE CONTROL 
        Isso resolve o problema das "infinitas soluções" escolhendo a mais próxima do zero.
        """
        f_end = self.funcs_fk_all_links[-1]
        args_0 = self._build_args(q_curr, np.zeros(self.num_dof))
        curr_pos = np.array(f_end(*args_0)).flatten()
        
        # Erro Cartesiano (Principal)
        error = target_pos - curr_pos
        
        # Jacobiano Linear
        J_num = np.array(self.func_J(*args_0))
        J_pos = J_num[:3, :] 
        
        # Damped Least Squares (Inversa Robusta)
        lambda_dls = 0.1 
        # J_inv = J.T * inv(J*J.T + lambda^2*I)
        # Para evitar inversão pesada 6x6, usamos a identidade do lado menor (3x3)
        J_dls_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_dls**2 * np.eye(3))
        
        # 1. Velocidade da Tarefa Principal (Chegar no XYZ)
        Kp_ik = 5.0
        dq_task = J_dls_pinv @ (error * Kp_ik)
        
        # 2. Controle de Espaço Nulo (Secondary Task)
        # Puxa as juntas para 'q_home' (zero) para evitar wind-up
        # Fórmula: dq_null = (I - J_pinv * J) * K * (q_home - q)
        I = np.eye(self.num_dof)
        
        # Definimos 'Home' como tudo zero (ou a posição inicial de conforto)
        q_home = np.zeros(self.num_dof) 
        
        # Erro em relação ao conforto (normalizado para menor caminho)
        q_err_null = self._wrap_to_pi(q_home - q_curr)
        
        # Ganho do Null Space (Menor que a tarefa principal)
        Kp_null = 1.0 
        
        # Projeção no Null Space
        null_projection = (I - J_dls_pinv @ J_pos)
        dq_null = null_projection @ (Kp_null * q_err_null)
        
        # Velocidade Final Combinada
        dq_total = dq_task + dq_null
        
        # Clamp de segurança
        dq_total = np.clip(dq_total, -3.0, 3.0) 
        
        q_next = q_curr + dq_total * dt
        return q_next, dq_total, np.zeros_like(dq_total)

    def run(self, t_total, Pi_list, Pf_list, Kp_val):
        dt_physics = 0.001  # 1000 Hz
        dt_visual  = 0.05   # 20 Hz
        
        steps_visual = int(t_total / dt_visual)
        substeps = int(dt_visual / dt_physics) 
        
        Pi = np.array(Pi_list, dtype=float)
        Pf = np.array(Pf_list, dtype=float)
        
        res_time = np.linspace(0, t_total, steps_visual)
        res_q = np.zeros((steps_visual, self.num_dof))
        res_tau = np.zeros((steps_visual, self.num_dof))
        anim_data = []

        q = np.zeros(self.num_dof) 
        dq = np.zeros(self.num_dof)
        
        # Ganhos PID
        KP = Kp_val * np.eye(self.num_dof)
        KD = 2 * np.sqrt(Kp_val) * np.eye(self.num_dof) 
        
        print(f"Simulando com Null Space Control...")
        
        current_time = 0.0
        
        for i in range(steps_visual):
            for _ in range(substeps):
                current_time += dt_physics
                
                # 1. Planejamento
                P_ref, V_ref, A_ref = self.trajectory_planning(current_time, t_total, Pi, Pf)
                
                # 2. IK Numérica (Com Null Space para evitar voltas)
                q_d, dq_d, ddq_d = self.solve_ik_numerical(P_ref, q, dt_physics)
                
                # 3. Dinâmica
                args = self._build_args(q, dq)
                try:
                    M = np.array(self.func_M(*args)).astype(np.float64)
                    C = np.array(self.func_C(*args)).flatten().astype(np.float64)
                    G = np.array(self.func_G(*args)).flatten().astype(np.float64)
                except Exception as e:
                    print("Erro numérico nas matrizes.")
                    raise e

                # 4. Controle (Com Wrap to Pi para menor caminho)
                e_pid_raw = q_d - q
                e_pid = self._wrap_to_pi(e_pid_raw) # <--- O SEGREDO DO CAMINHO CURTO
                
                e_dot = dq_d - dq
                
                u = ddq_d + (KD @ e_dot) + (KP @ e_pid)
                tau = M @ u + C + G
                
                # 5. Planta
                rhs = tau - C - G
                ddq = np.linalg.solve(M, rhs)
                
                q = q + dq * dt_physics
                dq = dq + ddq * dt_physics
                
                # Normaliza o próprio q para visualização limpa (Opcional, mas bom)
                # q = self._wrap_to_pi(q) 

            res_q[i, :] = e_pid
            res_tau[i, :] = tau
            
            # FK para animação
            links_pose = []
            links_pose.append([0,0,0])
            args_vis = self._build_args(q, dq)
            for f_fk in self.funcs_fk_all_links:
                pos = np.array(f_fk(*args_vis)).flatten()
                links_pose.append(list(pos))
            anim_data.append(links_pose)
            
        return res_time, res_q, res_tau, anim_data