import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sympy as sp

_WORKER_FUNCS = {}


class Decentralized_LADRC:
    """LADRC descentralizado para múltiplas juntas (2ª ordem)."""

    def __init__(self, num_dof, b0, wo, kp, kd, dt, z_limit=None, tau_limit=None,
                 max_wo_dt=0.1, z3_filter_alpha=1.0):
        self.num_dof = num_dof
        self.b0 = np.full(num_dof, b0, dtype=float) if np.isscalar(b0) else np.array(b0, dtype=float)
        self.wo = np.full(num_dof, wo, dtype=float) if np.isscalar(wo) else np.array(wo, dtype=float)
        self.kp = np.full(num_dof, kp, dtype=float) if np.isscalar(kp) else np.array(kp, dtype=float)
        self.kd = np.full(num_dof, kd, dtype=float) if np.isscalar(kd) else np.array(kd, dtype=float)
        self.dt = dt
        self.z_limit = z_limit
        self.tau_limit = tau_limit
        self.max_wo_dt = max_wo_dt
        # Filtro IIR de 1ª ordem na estimativa de distúrbio z3.
        # Reduz conteúdo de alta frequência que a lei de controle amplificaria.
        # 1.0 = sem filtro; valores próximos de 0 = filtro mais agressivo.
        self.z3_filter_alpha = float(np.clip(z3_filter_alpha, 0.0, 1.0))
        self.z3_filtered = None  # inicializado em reset_state

        self._refresh_gains()

        self.z1 = np.zeros(num_dof, dtype=float)
        self.z2 = np.zeros(num_dof, dtype=float)
        self.z3 = np.zeros(num_dof, dtype=float)

    def _refresh_gains(self):
        if self.max_wo_dt is not None:
            wo_dt = self.wo * self.dt
            limit_mask = wo_dt > self.max_wo_dt
            if np.any(limit_mask):
                self.wo = np.where(limit_mask, self.max_wo_dt / self.dt, self.wo)
        self.beta1 = 3.0 * self.wo
        self.beta2 = 3.0 * (self.wo ** 2)
        self.beta3 = self.wo ** 3

    def reset_state(self, q, dq, z3=None):
        self.z1 = np.array(q, dtype=float).copy()
        self.z2 = np.array(dq, dtype=float).copy()
        if z3 is None:
            self.z3 = np.zeros(self.num_dof, dtype=float)
        else:
            self.z3 = np.full(self.num_dof, z3, dtype=float) if np.isscalar(z3) else np.array(z3, dtype=float)
        self.z3_filtered = self.z3.copy()
        self._clip_states()

    def _clip_states(self):
        if self.z_limit is None:
            return
        self.z1 = np.clip(self.z1, -self.z_limit, self.z_limit)
        self.z2 = np.clip(self.z2, -self.z_limit, self.z_limit)
        self.z3 = np.clip(self.z3, -self.z_limit, self.z_limit)

    def update_eso(self, q, tau_prev):
        error = self.z1 - q
        z1_dot = self.z2 - self.beta1 * error
        z2_dot = self.z3 - self.beta2 * error + self.b0 * tau_prev
        z3_dot = -self.beta3 * error

        self.z1 += z1_dot * self.dt
        self.z2 += z2_dot * self.dt
        self.z3 += z3_dot * self.dt
        self._clip_states()

        # Filtro IIR em z3: atenua conteúdo de alta frequência na estimativa
        # de distúrbio antes de ela chegar à lei de controle.
        if self.z3_filtered is None:
            self.z3_filtered = self.z3.copy()
        else:
            a = self.z3_filter_alpha
            self.z3_filtered = a * self.z3 + (1.0 - a) * self.z3_filtered

    def compute_control(self, q_d, dq_d, ddq_d, q_meas=None, dq_meas=None):
        """Calcula o torque de controle.

        Quando q_meas/dq_meas são fornecidos, o PD usa as medições reais
        (mais limpo) e o ESO contribui apenas com z3 (cancelamento de
        distúrbio). Isso desacopla o ruído interno de z2 do sinal de controle.
        """
        if q_meas is not None and dq_meas is not None:
            q_fb  = q_meas
            dq_fb = dq_meas
        else:
            q_fb  = self.z1
            dq_fb = self.z2

        z3_use = self.z3_filtered if self.z3_filtered is not None else self.z3
        u0 = ddq_d + self.kp * (q_d - q_fb) + self.kd * (dq_d - dq_fb)
        b0_safe = np.where(self.b0 == 0.0, 1e-6, self.b0)
        u = (u0 - z3_use) / b0_safe
        if self.tau_limit is not None:
            u = np.clip(u, -self.tau_limit, self.tau_limit)
        return u


class SlidingModeControl:
    """Controlador SMC descentralizado com camada limite."""

    def __init__(self, num_dof, lambda_gain, K, phi):
        self.num_dof = num_dof
        self.lambda_gain = (
            np.full(num_dof, lambda_gain, dtype=float)
            if np.isscalar(lambda_gain)
            else np.array(lambda_gain, dtype=float)
        )
        self.K = np.full(num_dof, K, dtype=float) if np.isscalar(K) else np.array(K, dtype=float)
        self.phi = np.full(num_dof, phi, dtype=float) if np.isscalar(phi) else np.array(phi, dtype=float)
        self.phi = np.where(self.phi == 0, 1e-6, self.phi)

    def compute_tau(self, q, dq, q_d, dq_d, ddq_d):
        e = q - q_d
        e_dot = dq - dq_d
        S = e_dot + self.lambda_gain * e
        u_eq = ddq_d - self.lambda_gain * e_dot
        u_sw = self.K * np.tanh(S / self.phi)
        return u_eq - u_sw


def _init_worker(sym_vars, expr_M, expr_C, expr_G):
    global _WORKER_FUNCS
    _WORKER_FUNCS = {
        "M": sp.lambdify(sym_vars, expr_M, modules="numpy"),
        "C": sp.lambdify(sym_vars, expr_C, modules="numpy"),
        "G": sp.lambdify(sym_vars, expr_G, modules="numpy"),
    }


def _eval_worker(task):
    func_name, args = task
    return _WORKER_FUNCS[func_name](*args)


class RobotSimulator:
    def __init__(self, robot_math_instance, mode="Air"):
        self.bot = robot_math_instance
        self.mode = mode
        self.num_dof = len(self.bot.q)
        self.params_values = {} 
        self.q_home = np.zeros(self.num_dof)
        self.last_converged_q = None
        self.J_prev = None
        # Executor persistente: inicializado uma vez após a compilação e
        # reutilizado em todos os runs subsequentes, eliminando o custo de
        # respawn de processos e re-lambdify a cada simulação.
        self._executor = None
        self._executor_workers = 0

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
        self.expr_M = self.bot.M
        self.expr_C = self.bot.C_total
        self.expr_G = self.bot.G_vec
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

    def close(self):
        """Encerra o executor paralelo (chamar ao descartar o simulador)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._executor_workers = 0

    def warm_up_workers(self, max_workers=None):
        """Pré-inicializa o executor paralelo em background logo após a
        compilação do modelo, para que o primeiro run() não pague o custo
        de spawn + lambdify nas workers.
        """
        worker_count = max_workers or max(1, min(3, os.cpu_count() or 1))
        if self._executor is None or self._executor_workers != worker_count:
            self.close()
            self._executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(self.sym_vars, self.expr_M, self.expr_C, self.expr_G),
            )
            self._executor_workers = worker_count
            print(f"[Parallel] Workers pré-aquecidos ({worker_count} processos).")

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
        dq_limit=3.0,
        use_feedforward_vel=True,
    ):
        """ 
        Cinemática Inversa Numérica com Feedforward de Aceleração CORRIGIDO.
        Inclui compensação do termo de drift do Jacobiano (J_dot * dq).
        """
        f_end = self.funcs_fk_all_links[-1]
        args_0 = self._build_args(q_curr, np.zeros(self.num_dof))
        curr_pos = np.array(f_end(*args_0)).flatten()
        
        # Erro de Posição
        error = target_pos - curr_pos
        
        # Jacobiano Atual
        J_num = np.array(self.func_J(*args_0))
        J_pos = J_num[:3, :] 
        
        # --- CORREÇÃO: CÁLCULO NUMÉRICO DE J_DOT ---
        if self.J_prev is None:
            self.J_prev = J_pos.copy()
            J_dot = np.zeros_like(J_pos)
        else:
            # Derivada numérica finita: (J_curr - J_prev) / dt
            J_dot = (J_pos - self.J_prev) / dt
            self.J_prev = J_pos.copy()
            
        # Termo de Drift (Coriolis Cinemático): J_dot * dq
        # Isso diz: "Quanto a ponta se moveria só pela mudança da geometria?"
        drift_acc = J_dot @ dq_curr
        # -------------------------------------------

        # Damped Least Squares
        J_dls_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_dls**2 * np.eye(3))
        
        # Feedforward de Velocidade
        vel_ff = target_vel if use_feedforward_vel else np.zeros_like(target_vel)
        acc_ff = target_acc if use_feedforward_vel else np.zeros_like(target_acc)
        
        v_command = vel_ff + (error * Kp_ik)
        dq_task = J_dls_pinv @ v_command

        v_curr = J_pos @ dq_curr
        v_error = vel_ff - v_curr
        Kd_ik = 2.0 * np.sqrt(Kp_ik)
        
        # Aceleração Comandada no Espaço Cartesiano
        a_cartesian_target = acc_ff + (Kd_ik * v_error)
        
        # --- CORREÇÃO FINAL NA FÓRMULA DE ACELERAÇÃO ---
        # ddq = pinv(J) * ( a_cartesian - J_dot*dq )
        ddq_task = J_dls_pinv @ (a_cartesian_target - drift_acc)
        
        # Controle de Espaço Nulo
        I = np.eye(self.num_dof)
        q_target_null = self.q_home if hasattr(self, 'q_home') else np.zeros(self.num_dof)
        q_err_null = self._wrap_to_pi(q_target_null - q_curr)
        
        Kp_null = 1.0 
        null_projection = (I - J_dls_pinv @ J_pos)
        dq_null = null_projection @ (Kp_null * q_err_null)
        
        dq_total = dq_task + dq_null
        if dq_limit > 0:
            dq_total = dq_limit * np.tanh(dq_total / dq_limit)
        
        q_next = q_curr + dq_total * dt
        
        return q_next, dq_total, ddq_task
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
            dt_physics=None, dt_visual=None, init_at_start=True, q_init=None, zeta=1.0,
            dq_limit=3.0, use_feedforward_vel=True, use_parallel=False, max_workers=None,
            ctrl_params=None, disturbance_torque=None):
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
        if dq_limit < 0:
            raise ValueError("dq_limit deve ser maior ou igual a zero.")
        KP = Kp_val * np.eye(self.num_dof)
        zeta = getattr(self, "zeta", 1.0)
        KD = 2 * zeta * np.sqrt(Kp_val) * np.eye(self.num_dof)
        
        # Arrays de resultado
        res_time = np.linspace(0, t_total, steps_visual)
        res_q = np.zeros((steps_visual, self.num_dof))
        res_tau = np.zeros((steps_visual, self.num_dof))
        anim_data = []

        current_time = 0.0
        ctrl_params = ctrl_params or {}
        controller_type = ctrl_params.get("type", "Torque Computado")
        controller_type = controller_type.lower()
        ladrc = None
        smc = None
        tau_prev = np.zeros(self.num_dof)
        # Histórico apenas da saída do controlador (sem perturbação externa),
        # usado exclusivamente para atualizar o ESO do ADRC.
        tau_ctrl_prev = np.zeros(self.num_dof)
        if controller_type in {"adrc", "ladrc"}:
            wo = ctrl_params.get("wo", 20.0)
            max_wo_dt = ctrl_params.get("max_wo_dt", 0.1)
            if max_wo_dt is not None and wo * dt_physics > max_wo_dt:
                wo = max_wo_dt / dt_physics
                print("⚠️ wo ajustado para manter estabilidade numérica do ESO.")

            # Calcula M na postura inicial — reutilizado para b0 e z3_init.
            _args0 = self._build_args(q, np.zeros(self.num_dof))
            _M0 = np.array(self.func_M(*_args0)).astype(np.float64)
            _G0 = np.array(self.func_G(*_args0)).flatten().astype(np.float64)

            # Estimativa automática de b0 a partir da diagonal de M.
            # b0_i ≈ 1/M_ii é o ganho real de torque→aceleração por junta.
            # Um b0 errado injeta oscilações em z2 via "z2_dot += b0*tau"
            # e causa chattering quando |tau| satura.
            auto_b0 = ctrl_params.get("auto_b0", False)
            if auto_b0:
                M_diag = np.diag(_M0)
                b0_val = 1.0 / np.maximum(M_diag, 1e-4)
                print(f"[ADRC] b0 auto-estimado (1/M_ii): {np.round(b0_val, 4)}")
            else:
                b0_val = ctrl_params.get("b0", 1.0)

            ladrc = Decentralized_LADRC(
                num_dof=self.num_dof,
                b0=b0_val,
                wo=wo,
                kp=ctrl_params.get("kp", Kp_val),
                kd=ctrl_params.get("kd", 2 * zeta * np.sqrt(Kp_val)),
                dt=dt_physics,
                z_limit=ctrl_params.get("z_limit", 100.0),
                tau_limit=ctrl_params.get("tau_limit", 50.0),
                max_wo_dt=max_wo_dt,
                z3_filter_alpha=ctrl_params.get("z3_filter_alpha", 0.2),
            )
            # Com G feedforward ativo, z₃ só precisa estimar o distúrbio
            # residual (Coriolis + erros de modelo), que começa em zero.
            # Sem G feedforward (água com empuxo ≈ peso), inicializar z₃ com
            # -M⁻¹G ajuda a evitar o transitório de gravidade residual.
            use_gravity_ff = ctrl_params.get("gravity_ff", True)
            if use_gravity_ff:
                z3_init = np.zeros(self.num_dof)
            else:
                z3_init = -np.linalg.solve(_M0, _G0)
            ladrc.reset_state(q, dq, z3=z3_init)
        elif controller_type in {"smc", "slidingmode", "slidingmodecontrol"}:
            smc = SlidingModeControl(
                num_dof=self.num_dof,
                lambda_gain=ctrl_params.get("lambda", 5.0),
                K=ctrl_params.get("K", 5.0),
                phi=ctrl_params.get("phi", 0.1),
            )

        def _run_steps(executor):
            nonlocal current_time, q, dq, tau_prev, tau_ctrl_prev
            for i in range(steps_visual):
                for _ in range(substeps):
                    current_time += dt_physics

                    # --- AQUI: CHAMADA DINÂMICA DO PLANEJADOR ---
                    P_ref, V_ref, A_ref = self.trajectory_planning(
                        current_time, t_total, Pi, Pf,
                        mode=traj_mode, params=traj_params
                    )

                    # O Resto do loop físico continua IDÊNTICO ao que você já tinha...
                    # (IK Numérica, Dinâmica M/C/G, PID, Integração, etc)
                    q_d, dq_d, ddq_d = self.solve_ik_numerical(
                        P_ref,
                        V_ref,
                        A_ref,
                        q,
                        dq,
                        dt_physics,
                        dq_limit=dq_limit,
                        use_feedforward_vel=use_feedforward_vel,
                    )
                    args = self._build_args(q, dq)
                    if executor is None:
                        M = np.array(self.func_M(*args)).astype(np.float64)
                        C = np.array(self.func_C(*args)).flatten().astype(np.float64)
                        G = np.array(self.func_G(*args)).flatten().astype(np.float64)
                    else:
                        futures = {
                            "M": executor.submit(_eval_worker, ("M", args)),
                            "C": executor.submit(_eval_worker, ("C", args)),
                            "G": executor.submit(_eval_worker, ("G", args)),
                        }
                        M = np.array(futures["M"].result()).astype(np.float64)
                        C = np.array(futures["C"].result()).flatten().astype(np.float64)
                        G = np.array(futures["G"].result()).flatten().astype(np.float64)

                    e_pid = self._wrap_to_pi(q_d - q)
                    e_dot = dq_d - dq

                    if ladrc is not None:
                        # ESO recebe apenas a saída do controlador ADRC (sem G
                        # feedforward nem perturbação externa) para que z₃ estime
                        # apenas o distúrbio residual (Coriolis + erros de modelo).
                        ladrc.update_eso(q, tau_ctrl_prev)
                        # PD usa medições reais (q, dq); ESO contribui com z3.
                        u_adrc = ladrc.compute_control(q_d, dq_d, ddq_d,
                                                       q_meas=q, dq_meas=dq)
                        tau_filter_alpha = ctrl_params.get("tau_filter_alpha", 1.0)
                        tau_filter_alpha = np.clip(tau_filter_alpha, 0.0, 1.0)
                        u_adrc = tau_ctrl_prev + tau_filter_alpha * (u_adrc - tau_ctrl_prev)

                        # Feedforward explícito de G(q) e C(q,dq): remove os
                        # termos conhecidos da dinâmica do que z₃ precisa estimar.
                        # → G(q):    crítico no ar (gravity ≠ 0); inócuo na água (G≈0)
                        # → C(q,dq): elimina a oscilação crescente no início da
                        #            trajetória, pois C ∝ dq² acompanha o perfil de
                        #            velocidade cúbico. Sem este FF, z₃ persegue C em
                        #            crescimento com modos discretos oscilatórios.
                        # Com ambos ativos, z₃ só estima distúrbios externos e erros
                        # de modelo — sinais pequenos e quase constantes.
                        use_gravity_ff   = ctrl_params.get("gravity_ff",   True)
                        use_coriolis_ff  = ctrl_params.get("coriolis_ff",  True)
                        u_control = (u_adrc
                                     + (G if use_gravity_ff  else 0.0)
                                     + (C if use_coriolis_ff else 0.0))
                    elif smc is not None:
                        u_control = smc.compute_tau(q, dq, q_d, dq_d, ddq_d)
                    else:
                        u = ddq_d + (KD @ e_dot) + (KP @ e_pid)
                        u_control = M @ u + C + G

                    tau_disturbance = np.zeros(self.num_dof)
                    if disturbance_torque is not None and self.num_dof >= 2:
                        tau_disturbance[1] = disturbance_torque
                    tau_applied = u_control + tau_disturbance
                    # ESO recebe somente a saída ADRC pura (u_adrc), sem G
                    # feedforward nem perturbação, para que z₃ estime apenas
                    # os distúrbios residuais que o controlador não conhece.
                    tau_ctrl_prev = u_adrc if ladrc is not None else u_control
                    tau_prev = tau_applied

                    ddq = np.linalg.solve(M, tau_applied - C - G)
                    q += dq * dt_physics
                    dq += ddq * dt_physics
                    q = self._wrap_to_pi(q) # Wrap essencial
                    if not (np.isfinite(q).all() and np.isfinite(dq).all()):
                        raise FloatingPointError("Estado não finito detectado durante a simulação.")

                res_q[i,:] = e_pid
                res_tau[i,:] = tau_applied

                # FK para animação
                links_pose = [[0,0,0]]
                args_vis = self._build_args(q, dq)
                for f_fk in self.funcs_fk_all_links:
                    pos = np.array(f_fk(*args_vis)).flatten()
                    if not np.isfinite(pos).all():
                        raise FloatingPointError(
                            "FK produziu valores não finitos durante a simulação."
                        )
                    links_pose.append(list(pos))
                anim_data.append(links_pose)

        if use_parallel:
            worker_count = max_workers or max(1, min(3, os.cpu_count() or 1))
            # Reutiliza o executor se já existir com o mesmo número de workers.
            # Isso elimina: respawn de processos + re-lambdify nas workers
            # a cada run (custo era de 2–8 s por simulação).
            if self._executor is None or self._executor_workers != worker_count:
                self.close()  # derruba executor anterior, se houver
                self._executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_worker,
                    initargs=(self.sym_vars, self.expr_M, self.expr_C, self.expr_G),
                )
                self._executor_workers = worker_count
                print(f"[Parallel] Executor inicializado com {worker_count} workers.")
            _run_steps(self._executor)
        else:
            _run_steps(None)

        return res_time, res_q, res_tau, anim_data
