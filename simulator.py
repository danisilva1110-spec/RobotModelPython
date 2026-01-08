import numpy as np
import sympy as sp

class RobotSimulator:
    def __init__(self, robot_math_instance, mode="Air"):
        self.bot = robot_math_instance
        self.mode = mode
        self.num_dof = len(self.bot.q)
        self.params_values = {} 
        self.q_home = np.zeros(self.num_dof)
        self.last_converged_q = None

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

    def solve_ik_numerical(
        self,
        target_pos,
        target_vel,
        target_acc,
        q_curr,
        dq_curr,
        dt,
        Kp_ik=5.0,
        lambda_dls=0.1,
        ik_pos_iters=3,
        ik_pos_step=0.5,
    ):
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
        J_dls_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_dls**2 * np.eye(3))
        
        # Antes era apenas: dq_task = J_dls_pinv @ (error * Kp_ik)
        # AGORA somamos a velocidade desejada (target_vel) vinda do planejador
        # Isso é o Feedforward: O robô já se move na velocidade da curva mesmo se o erro for zero.
        v_command = target_vel + (error * Kp_ik)
        
        dq_task = J_dls_pinv @ v_command

        v_curr = J_pos @ dq_curr
        v_error = target_vel - v_curr
        Kd_ik = 2.0 * np.sqrt(Kp_ik)
        a_command = target_acc + (Kd_ik * v_error) + (Kp_ik * error)
        ddq_task = J_dls_pinv @ a_command
        
        # Controle de Espaço Nulo (Mantém igual)
        I = np.eye(self.num_dof)
        
        # Usa o q_home da classe se existir, senão zero
        q_target_null = self.q_home if hasattr(self, 'q_home') else np.zeros(self.num_dof)
        q_err_null = self._wrap_to_pi(q_target_null - q_curr)
        
        Kp_null = 1.0 
        null_projection = (I - J_dls_pinv @ J_pos)
        dq_null = null_projection @ (Kp_null * q_err_null)
        
        dq_total = dq_task + dq_null
        dq_total = np.clip(dq_total, -3.0, 3.0)

        # IK de posição (algumas iterações DLS) diretamente no P_ref atual
        q_d = np.array(q_curr, dtype=float).copy()
        for _ in range(max(1, ik_pos_iters)):
            args_d = self._build_args(q_d, np.zeros(self.num_dof))
            curr_pos_d = np.array(f_end(*args_d)).flatten()
            pos_error = target_pos - curr_pos_d
            if np.linalg.norm(pos_error) < 1e-4:
                break
            J_num_d = np.array(self.func_J(*args_d))
            J_pos_d = J_num_d[:3, :]
            J_dls_pinv_d = J_pos_d.T @ np.linalg.inv(
                J_pos_d @ J_pos_d.T + lambda_dls**2 * np.eye(3)
            )
            dq_pos = J_dls_pinv_d @ pos_error
            q_d = self._wrap_to_pi(q_d + ik_pos_step * dq_pos)

        # Retornamos q_d, dq_total e ddq_task para usar na dinâmica
        return q_d, dq_total, ddq_task

    def solve_ik_initial(
        self,
        target_pos,
        q_init,
        max_iters=200,
        tol=1e-3,
        lambda_init=0.2,
        min_step=1e-4,
    ):
        """
        IK inicial mais robusta (Levenberg-Marquardt + line search).
        Retorna (q_final, convergiu, erro_final, iteracoes).
        """
        f_end = self.funcs_fk_all_links[-1]
        q_curr = np.array(q_init, dtype=float).copy()
        lambda_dls = lambda_init
        last_error = np.inf
        stall_count = 0

        for i in range(max_iters):
            args = self._build_args(q_curr, np.zeros(self.num_dof))
            curr_pos = np.array(f_end(*args)).flatten()
            error = target_pos - curr_pos
            error_norm = np.linalg.norm(error)

            if error_norm < tol:
                return q_curr, True, error_norm, i + 1

            J_num = np.array(self.func_J(*args))
            J_pos = J_num[:3, :]
            J_dls_pinv = J_pos.T @ np.linalg.inv(
                J_pos @ J_pos.T + (lambda_dls**2) * np.eye(3)
            )
            dq = J_dls_pinv @ error

            # Line search: reduz passo até melhorar o erro
            alpha = 1.0
            improved = False
            while alpha >= min_step:
                q_next = self._wrap_to_pi(q_curr + alpha * dq)
                args_next = self._build_args(q_next, np.zeros(self.num_dof))
                next_pos = np.array(f_end(*args_next)).flatten()
                next_error = target_pos - next_pos
                next_error_norm = np.linalg.norm(next_error)
                if next_error_norm < error_norm:
                    q_curr = q_next
                    error_norm = next_error_norm
                    improved = True
                    break
                alpha *= 0.5

            if not improved:
                lambda_dls = min(10.0, lambda_dls * 1.5)
            else:
                lambda_dls = max(1e-4, lambda_dls * 0.9)

            if abs(last_error - error_norm) < tol * 0.1:
                stall_count += 1
            else:
                stall_count = 0
            last_error = error_norm

            if stall_count >= 10:
                break

        return q_curr, False, last_error, max_iters

    def run(self, t_total, Pi_list, Pf_list, Kp_val, traj_mode="Line", traj_params=None,
            dt_physics=None, dt_visual=None, init_at_start=True, q_init=None, zeta=1.0):
        # ... (Início igual ao original) ...
        dt_physics = 0.001 if dt_physics is None else dt_physics
        dt_visual = 0.05 if dt_visual is None else dt_visual

        if dt_physics <= 0 or dt_visual <= 0:
            raise ValueError("dt_physics e dt_visual devem ser maiores que zero.")

        if dt_physics > 0.01 or dt_visual > 0.1:
            print("⚠️ Passos de integração grandes podem causar instabilidade numérica.")

        substeps = max(1, int(np.ceil(dt_visual / dt_physics)))
        dt_visual_effective = dt_physics * substeps
        steps_visual = max(1, int(np.ceil(t_total / dt_visual_effective)))
        self.last_dt_visual = dt_visual_effective
        
        Pi = np.array(Pi_list, dtype=float)
        Pf = np.array(Pf_list, dtype=float)
        
        # Inicialização (Com postura preferida se definida)
        q_home = np.copy(self.q_home) if hasattr(self, 'q_home') else np.zeros(self.num_dof)
        q = np.copy(q_home)
        dq = np.zeros(self.num_dof)

        if init_at_start:
            if q_init is None:
                q_init = np.copy(self.last_converged_q) if self.last_converged_q is not None else np.copy(q_home)
            else:
                q_init = np.array(q_init, dtype=float).copy()
            q_init, converged, init_error, init_iters = self.solve_ik_initial(
                target_pos=Pi,
                q_init=q_init,
                max_iters=300,
                tol=1e-3,
                lambda_init=0.2,
            )
            if converged:
                q = q_init
                dq = np.zeros(self.num_dof)
                self.last_converged_q = np.copy(q_init)
            else:
                print(
                    "⚠️ IK inicial não convergiu. "
                    f"Erro final {init_error:.3e} após {init_iters} iterações. "
                    "Usando postura home como inicial."
                )
                q = np.copy(q_home)
                dq = np.zeros(self.num_dof)
        
        if zeta <= 0:
            raise ValueError("zeta deve ser maior que zero.")
        self.zeta = zeta
        # Ganhos PD (em aceleração)
        if Kp_val <= 0:
            raise ValueError("Kp deve ser maior que zero.")
        KP = Kp_val * np.eye(self.num_dof)
        zeta = getattr(self, "zeta", 1.0)
        KD = 2 * zeta * np.sqrt(Kp_val) * np.eye(self.num_dof)
        
        # Arrays de resultado
        res_time = np.linspace(0, t_total, steps_visual)
        res_q = np.zeros((steps_visual, self.num_dof))
        res_tau = np.zeros((steps_visual, self.num_dof))
        anim_data = []

        current_time = 0.0
        
        for i in range(steps_visual):
            for _ in range(substeps):
                current_time += dt_physics
                
                # --- AQUI: CHAMADA DINÂMICA DO PLANEJADOR ---
                P_ref, V_ref, A_ref = self.trajectory_planning(
                    current_time, t_total, Pi, Pf, 
                    mode=traj_mode, params=traj_params
                )
                
                # IK Numérica (q_d via DLS no P_ref atual, sem integrar q_curr + dq*dt)
                q_d, dq_d, ddq_d = self.solve_ik_numerical(
                    P_ref,
                    V_ref,
                    A_ref,
                    q,
                    dq,
                    dt_physics,
                )
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
