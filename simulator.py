import numpy as np
import sympy as sp

class RobotSimulator:
    def __init__(self, robot_math_instance, mode="Air"):
        self.bot = robot_math_instance
        self.mode = mode
        self.num_dof = len(self.bot.q)
        self.params_values = {} 
        self.q_home = np.zeros(self.num_dof)

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

    def set_q_home(self, q_home):
        q_home = np.array(q_home, dtype=float)
        self.q_home = self._wrap_to_pi(q_home)

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

    def trajectory_planning(self, t, t_total, Pi, Pf, mode="Line", params=None):
        """ Implementação fiel do algoritmo MATLAB 'Planejamentos.txt' """
        if t >= t_total: return Pf, np.zeros(3), np.zeros(3)
        
        # Polinômio Cúbico (s, sd, sdd) - Igual ao FCubica do MATLAB
        tau = t / t_total
        s = 3*(tau**2) - 2*(tau**3)
        sd = (6*tau - 6*(tau**2)) / t_total
        sdd = (6 - 12*tau) / (t_total**2)

        if mode == "Line":
            d = Pf - Pi
            return (Pi + d*s), (d*sd), (d*sdd)

        elif mode == "Circle":
            # Parâmetros vindos da Interface
            R = params.get('radius', 0.2)
            normal = np.array(params.get('normal', [1,0,0]), dtype=float)
            normal = normal / np.linalg.norm(normal)
            sentido = params.get('direction', 1) # 1 ou -1

            # Lógica Vetorial (Tradução direta do seu .txt)
            v = Pf - Pi
            d_chord = np.linalg.norm(v)
            
            if d_chord > 2*R: R = d_chord/2 + 0.001 # Segurança

            mi = (Pi + Pf) / 2
            h = np.sqrt(max(0, R**2 - (d_chord/2)**2)) # max(0,...) evita erro numérico

            v_perp = np.cross(v, normal)
            if np.linalg.norm(v_perp) < 1e-6: # Proteção contra colinearidade
                 return self.trajectory_planning(t, t_total, Pi, Pf, mode="Line")
            
            v_perp = v_perp / np.linalg.norm(v_perp)

            # Centro C
            C = mi + h * v_perp if sentido > 0 else mi - h * v_perp

            # Bases do Plano (e1, e2)
            e1 = (Pi - C)
            e1 = e1 / np.linalg.norm(e1)
            
            e2 = np.cross(normal, e1)
            e2 = e2 / np.linalg.norm(e2)

            # "Garanta que e2 aponta na direção de Pi->Pf" (Do seu código)
            if np.dot(e2, v) < 0: e2 = -e2

            # Ângulos (Theta relativo a e1, então start é sempre 0)
            theta_start = 0.0
            vec_Pf = Pf - C
            theta_end = np.arctan2(np.dot(vec_Pf, e2), np.dot(vec_Pf, e1))

            # Ajuste de voltas (Unwrapping)
            if sentido > 0:
                if theta_end < theta_start: theta_end += 2*np.pi
            else:
                if theta_end > theta_start: theta_end -= 2*np.pi

            # Interpolação Angular
            theta_t = theta_start + (theta_end - theta_start) * s
            dtheta  = (theta_end - theta_start) * sd
            ddtheta = (theta_end - theta_start) * sdd

            # Cinemática Direta do Arco
            cos_th, sin_th = np.cos(theta_t), np.sin(theta_t)
            P = C + R * (cos_th * e1 + sin_th * e2)
            
            # V = dP/dt
            V = R * dtheta * (-sin_th * e1 + cos_th * e2)
            
            # A = dV/dt (Regra da cadeia + produto)
            tangent = (-sin_th * e1 + cos_th * e2)
            normal_vec = (-cos_th * e1 - sin_th * e2)
            A = R * (ddtheta * tangent + (dtheta**2) * normal_vec)

            return P, V, A

        return Pf, np.zeros(3), np.zeros(3)

    def solve_ik_numerical(self, target_pos, target_vel, q_curr, dt, use_nullspace=True, kp_ik=5.0):
        """ 
        Cinemática Inversa Numérica com Feedforward de Velocidade.
        Agora o robô 'sabe' a velocidade da curva, não só a posição.
        """
        f_end = self.funcs_fk_all_links[-1]
        args_0 = self._build_args(q_curr, np.zeros(self.num_dof))
        curr_pos = np.array(f_end(*args_0)).flatten()
        
        # Erro de Posição (Proporcional)
        error = target_pos - curr_pos
        
        # Jacobiano
        J_num = np.array(self.func_J(*args_0))
        J_pos = J_num[:3, :] 
        
        # Damped Least Squares
        lambda_dls = 0.1 
        J_dls_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_dls**2 * np.eye(3))
        
        # --- A CORREÇÃO MÁGICA AQUI ---
        Kp_ik = kp_ik
        
        # Antes era apenas: dq_task = J_dls_pinv @ (error * Kp_ik)
        # AGORA somamos a velocidade desejada (target_vel) vinda do planejador
        # Isso é o Feedforward: O robô já se move na velocidade da curva mesmo se o erro for zero.
        v_command = target_vel + (error * Kp_ik)
        
        dq_task = J_dls_pinv @ v_command

        # Regularização para penalizar variações grandes nas juntas
        dq_norm = np.linalg.norm(dq_task)
        reg_gain = 0.15
        if dq_norm > 0.0:
            dq_task = dq_task / (1.0 + reg_gain * dq_norm)

        # Custo de mínimo deslocamento (evita voltas longas)
        q_ref = q_curr + dq_task * dt
        q_err = self._wrap_to_pi(q_ref - q_curr)
        min_disp_gain = 0.5
        dq_min_disp = q_err / max(dt, 1e-6)
        dq_task = dq_task + min_disp_gain * (dq_min_disp - dq_task)
        
        # Controle de Espaço Nulo (Mantém igual)
        dq_total = dq_task
        if use_nullspace:
            I = np.eye(self.num_dof)

            # Usa o q_home da classe se existir, senão zero
            q_target_null = self.q_home if hasattr(self, 'q_home') else np.zeros(self.num_dof)
            q_err_null = self._wrap_to_pi(q_target_null - q_curr)

            Kp_null = 1.0
            null_projection = (I - J_dls_pinv @ J_pos)
            dq_null = null_projection @ (Kp_null * q_err_null)

            dq_total = dq_task + dq_null
        dq_limit = 2.0
        dq_total = np.clip(dq_total, -dq_limit, dq_limit)
        
        q_next = q_curr + dq_total * dt
        q_next = self._wrap_to_pi(q_next)
        dq_total = self._wrap_to_pi(q_next - q_curr) / max(dt, 1e-6)
        
        # Retornamos dq_total para usar na dinâmica
        return q_next, dq_total, np.zeros_like(dq_total)

    def run(self, t_total, Pi_list, Pf_list, Kp_val, traj_mode="Line", traj_params=None):
        # ... (Início igual ao original) ...
        dt_physics = 0.001
        dt_visual  = 0.05
        pre_time = max(0.5, min(2.0, 0.2 * t_total))
        total_time = t_total + pre_time
        steps_visual = int(total_time / dt_visual)
        substeps = int(dt_visual / dt_physics)
        
        Pi = np.array(Pi_list, dtype=float)
        Pf = np.array(Pf_list, dtype=float)
        
        # Inicialização (Com postura preferida se definida)
        if hasattr(self, 'q_home'):
            self.q_home = self._wrap_to_pi(self.q_home)
            q = np.copy(self.q_home)
        else:
            q = np.zeros(self.num_dof)
        dq = np.zeros(self.num_dof)

        # Calcula posição atual do efetuador final para criar a reta até Pi
        f_end = self.funcs_fk_all_links[-1]
        args_0 = self._build_args(q, dq)
        start_pos = np.array(f_end(*args_0)).flatten()
        
        # Ganhos PID
        KP = Kp_val * np.eye(self.num_dof)
        KD = 2 * np.sqrt(Kp_val) * np.eye(self.num_dof)
        
        # Arrays de resultado
        res_time = np.linspace(0, total_time, steps_visual)
        res_q = np.zeros((steps_visual, self.num_dof))
        res_tau = np.zeros((steps_visual, self.num_dof))
        anim_data = []

        current_time = 0.0
        
        for i in range(steps_visual):
            for _ in range(substeps):
                current_time += dt_physics
                
                # --- AQUI: CHAMADA DINÂMICA DO PLANEJADOR ---
                if current_time <= pre_time:
                    P_ref, V_ref, A_ref = self.trajectory_planning(
                        current_time, pre_time, start_pos, Pi,
                        mode="Line", params=None
                    )
                    q_d, dq_d, ddq_d = self.solve_ik_numerical(
                        P_ref, V_ref, q, dt_physics, use_nullspace=False, kp_ik=10.0
                    )
                    q = q_d
                    dq = dq_d
                    e_pid = np.zeros(self.num_dof)
                    tau = np.zeros(self.num_dof)
                    continue
                else:
                    t_main = current_time - pre_time
                    P_ref, V_ref, A_ref = self.trajectory_planning(
                        t_main, t_total, Pi, Pf,
                        mode=traj_mode, params=traj_params
                    )
                
                # O Resto do loop físico continua IDÊNTICO ao que você já tinha...
                # (IK Numérica, Dinâmica M/C/G, PID, Integração, etc)
                q_d, dq_d, ddq_d = self.solve_ik_numerical(P_ref, V_ref, q, dt_physics)
                args = self._build_args(q, dq)
                M = np.array(self.func_M(*args)).astype(np.float64)
                C = np.array(self.func_C(*args)).flatten().astype(np.float64)
                G = np.array(self.func_G(*args)).flatten().astype(np.float64)
                
                e_pid = self._wrap_to_pi(q_d - q)
                e_dot = dq_d - dq
                u = ddq_d + (KD @ e_dot) + (KP @ e_pid)
                tau = M @ u + C + G
                
                ddq = np.linalg.solve(M, tau - C - G)
                q += dq * dt_physics
                dq += ddq * dt_physics
                q = self._wrap_to_pi(q) # Wrap essencial

            res_q[i,:] = e_pid
            res_tau[i,:] = tau
            
            # FK para animação
            links_pose = [[0,0,0]]
            args_vis = self._build_args(q, dq)
            for f_fk in self.funcs_fk_all_links:
                pos = np.array(f_fk(*args_vis)).flatten()
                links_pose.append(list(pos))
            anim_data.append(links_pose)
            
        return res_time, res_q, res_tau, anim_data
