import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sympy as sp

_WORKER_FUNCS = {}


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
        self.last_target_rot = None
        self.J_prev = None
        self.has_vehicle_base = False
        self.vehicle_dof_indices = []
        self.manipulator_dof_indices = []
        self._resolve_dof_groups()

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
        self.func_fk_rot = sp.lambdify(
            self.sym_vars,
            self.bot.frames[-1][:3, :3],
            modules='numpy',
        )

        self.funcs_fk_all_links = []
        for frame in self.bot.frames:
            pos_expr = frame[:3, 3] # Pega X, Y, Z
            f_fk = sp.lambdify(self.sym_vars, pos_expr, modules='numpy')
            self.funcs_fk_all_links.append(f_fk)

        print("Compilação concluída!")

    def _resolve_dof_groups(self):
        num_dof = self.num_dof
        has_vehicle_base = bool(getattr(self.bot, "has_vehicle_base", False))
        vehicle_dof_indices = list(
            getattr(self.bot, "vehicle_dof_indices", []) or []
        )
        manipulator_dof_indices = list(
            getattr(self.bot, "manipulator_dof_indices", []) or []
        )
        vehicle_dof_indices = [
            i for i in sorted(set(vehicle_dof_indices)) if 0 <= i < num_dof
        ]
        manipulator_dof_indices = [
            i for i in sorted(set(manipulator_dof_indices)) if 0 <= i < num_dof
        ]

        if not has_vehicle_base:
            vehicle_dof_indices = []
            manipulator_dof_indices = list(range(num_dof))
            print("ℹ️ Base móvel desativada; solver opera como robô fixo.")
        else:
            if not vehicle_dof_indices and not manipulator_dof_indices:
                vehicle_dof_indices = list(range(min(6, num_dof)))
                manipulator_dof_indices = [
                    i for i in range(num_dof) if i not in vehicle_dof_indices
                ]
                print(
                    "ℹ️ Base móvel ativa sem índices; usando fallback "
                    "para separar base/manipulador."
                )
            elif not manipulator_dof_indices:
                manipulator_dof_indices = [
                    i for i in range(num_dof) if i not in vehicle_dof_indices
                ]
            elif not vehicle_dof_indices:
                vehicle_dof_indices = [
                    i for i in range(num_dof) if i not in manipulator_dof_indices
                ]
            manipulator_dof_indices = [
                i for i in manipulator_dof_indices if i not in vehicle_dof_indices
            ]
            if not manipulator_dof_indices:
                manipulator_dof_indices = [
                    i for i in range(num_dof) if i not in vehicle_dof_indices
                ]
            if not vehicle_dof_indices:
                vehicle_dof_indices = [
                    i for i in range(num_dof) if i not in manipulator_dof_indices
                ]

        self.has_vehicle_base = has_vehicle_base
        self.vehicle_dof_indices = vehicle_dof_indices
        self.manipulator_dof_indices = manipulator_dof_indices

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

    def _orientation_error(self, target_rot, curr_rot):
        """Erro de orientação baseado em matriz de rotação (aprox. eixo-ângulo)."""
        rot_err = target_rot @ curr_rot.T
        return 0.5 * np.array(
            [
                rot_err[2, 1] - rot_err[1, 2],
                rot_err[0, 2] - rot_err[2, 0],
                rot_err[1, 0] - rot_err[0, 1],
            ]
        )

    def _normalize_vector(self, vec, eps=1e-8):
        norm = np.linalg.norm(vec)
        if norm < eps:
            return None
        return vec / norm

    def _rotation_from_z_axis(self, z_axis, up_ref=None):
        z_axis = self._normalize_vector(np.array(z_axis, dtype=float))
        if z_axis is None:
            return None
        up = np.array([0.0, 0.0, 1.0]) if up_ref is None else np.array(up_ref, dtype=float)
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(up, z_axis)
        x_axis = self._normalize_vector(x_axis)
        if x_axis is None:
            return None
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _rotation_from_x_axis(self, x_axis, up_ref=None):
        x_axis = self._normalize_vector(np.array(x_axis, dtype=float))
        if x_axis is None:
            return None
        up = np.array([0.0, 0.0, 1.0]) if up_ref is None else np.array(up_ref, dtype=float)
        y_axis = np.cross(up, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(up, x_axis)
        y_axis = self._normalize_vector(y_axis)
        if y_axis is None:
            return None
        z_axis = np.cross(x_axis, y_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _derive_orientation_reference(self, preset, target_pos, fallback_direction=None):
        if preset is None or preset == "Desligado":
            return None
        if preset == "Sempre para baixo":
            target_rot = self._rotation_from_z_axis([0.0, 0.0, -1.0])
            self.last_target_rot = target_rot
            return target_rot
        target_vec = np.array(target_pos, dtype=float)
        if np.linalg.norm(target_vec) < 1e-6:
            if self.last_target_rot is not None:
                return self.last_target_rot
            if fallback_direction is not None:
                target_vec = np.array(fallback_direction, dtype=float)
        direction = self._normalize_vector(target_vec)
        if direction is None:
            return None
        if preset == "Olhar para o alvo":
            target_rot = self._rotation_from_z_axis(direction)
            self.last_target_rot = target_rot
            return target_rot
        if preset == "Inspeção frontal":
            target_rot = self._rotation_from_x_axis(direction)
            self.last_target_rot = target_rot
            return target_rot
        return None

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
        target_rot=None,
        target_ang_vel=None,
        target_ang_acc=None,
        priority="position",
    ):
        """ 
        Cinemática Inversa Numérica com Feedforward de Aceleração CORRIGIDO.
        Inclui compensação do termo de drift do Jacobiano (J_dot * dq).
        """
        f_end = self.funcs_fk_all_links[-1]
        args_0 = self._build_args(q_curr, np.zeros(self.num_dof))
        curr_pos = np.array(f_end(*args_0)).flatten()
        curr_rot = np.array(self.func_fk_rot(*args_0))
        if target_rot is not None:
            target_rot = np.array(target_rot, dtype=float)
        
        # Erro de Posição
        error = target_pos - curr_pos
        orient_error = (
            self._orientation_error(target_rot, curr_rot)
            if target_rot is not None
            else np.zeros(3)
        )
        
        # Jacobiano Atual
        J_num = np.array(self.func_J(*args_0))
        J_pos = J_num[:3, :] 
        J_ang = J_num[3:6, :]
        
        # --- CORREÇÃO: CÁLCULO NUMÉRICO DE J_DOT ---
        if self.J_prev is None:
            self.J_prev = J_num.copy()
            J_dot = np.zeros_like(J_num)
        else:
            # Derivada numérica finita: (J_curr - J_prev) / dt
            J_dot = (J_num - self.J_prev) / dt
            self.J_prev = J_num.copy()
            
        # Termo de Drift (Coriolis Cinemático): J_dot * dq
        # Isso diz: "Quanto a ponta se moveria só pela mudança da geometria?"
        drift_acc_pos = J_dot[:3, :] @ dq_curr
        drift_acc_ang = J_dot[3:6, :] @ dq_curr
        # -------------------------------------------

        if self.has_vehicle_base and target_rot is not None:
            priority = "orientation"

        if target_rot is None:
            priority = "position"

        valid_priorities = {"position", "orientation", "balanced"}
        if priority not in valid_priorities:
            raise ValueError(
                f"Prioridade inválida ({priority}). Use uma de {valid_priorities}."
            )

        # Damped Least Squares
        J_dls_pinv = J_pos.T @ np.linalg.inv(
            J_pos @ J_pos.T + lambda_dls**2 * np.eye(3)
        )
        
        # Feedforward de Velocidade
        vel_ff = target_vel if use_feedforward_vel else np.zeros_like(target_vel)
        acc_ff = target_acc if use_feedforward_vel else np.zeros_like(target_acc)
        ang_vel_ff = (
            target_ang_vel if target_ang_vel is not None else np.zeros(3)
        )
        ang_acc_ff = (
            target_ang_acc if target_ang_acc is not None else np.zeros(3)
        )
        if not use_feedforward_vel:
            ang_vel_ff = np.zeros(3)
            ang_acc_ff = np.zeros(3)
        
        v_command = vel_ff + (error * Kp_ik)

        v_curr = J_pos @ dq_curr
        v_error = vel_ff - v_curr
        Kd_ik = 2.0 * np.sqrt(Kp_ik)
        
        # Aceleração Comandada no Espaço Cartesiano
        a_cartesian_target = acc_ff + (Kd_ik * v_error)
        w_curr = J_ang @ dq_curr
        w_error = ang_vel_ff - w_curr
        a_ang_target = ang_acc_ff + (Kd_ik * w_error)

        if priority == "position":
            if self.num_dof < 3 or np.linalg.matrix_rank(J_pos) < 3:
                print(
                    "⚠️ DOFs insuficientes para tarefa de posição. "
                    "Usando solução aproximada (DLS)."
                )
            dq_task = J_dls_pinv @ v_command
            # --- CORREÇÃO FINAL NA FÓRMULA DE ACELERAÇÃO ---
            # ddq = pinv(J) * ( a_cartesian - J_dot*dq )
            ddq_task = J_dls_pinv @ (a_cartesian_target - drift_acc_pos)
        elif priority == "orientation":
            orient_indices = list(range(self.num_dof))
            if self.has_vehicle_base and self.manipulator_dof_indices:
                orient_indices = self.manipulator_dof_indices
                J_ang_subset = J_ang[:, orient_indices]
                if self.num_dof < 3 or np.linalg.matrix_rank(J_ang_subset) < 3:
                    print(
                        "⚠️ DOFs insuficientes na orientação do manipulador. "
                        "Usando todos os DOFs disponíveis."
                    )
                    orient_indices = list(range(self.num_dof))
            J_ang_masked = np.zeros_like(J_ang)
            J_ang_masked[:, orient_indices] = J_ang[:, orient_indices]
            if self.num_dof < 3 or np.linalg.matrix_rank(J_ang_masked) < 3:
                print(
                    "⚠️ DOFs insuficientes para tarefa de orientação. "
                    "Usando solução aproximada (DLS)."
                )
            J_ang_pinv = J_ang_masked.T @ np.linalg.inv(
                J_ang_masked @ J_ang_masked.T + lambda_dls**2 * np.eye(3)
            )
            w_command = ang_vel_ff + (orient_error * Kp_ik)
            dq_orient = J_ang_pinv @ w_command
            ddq_orient = J_ang_pinv @ (a_ang_target - drift_acc_ang)

            null_projection_orient = np.eye(self.num_dof) - J_ang_pinv @ J_ang_masked
            J_pos_null = J_pos @ null_projection_orient
            if self.num_dof < 3 or np.linalg.matrix_rank(J_pos_null) < 3:
                print(
                    "⚠️ DOFs insuficientes para tarefa de posição no espaço nulo. "
                    "Usando solução aproximada (DLS)."
                )
            J_pos_null_pinv = J_pos_null.T @ np.linalg.inv(
                J_pos_null @ J_pos_null.T + lambda_dls**2 * np.eye(3)
            )
            dq_pos = J_pos_null_pinv @ (v_command - J_pos @ dq_orient)
            ddq_pos = J_pos_null_pinv @ (
                a_cartesian_target - drift_acc_pos - J_pos @ ddq_orient
            )
            dq_task = dq_orient + null_projection_orient @ dq_pos
            ddq_task = ddq_orient + null_projection_orient @ ddq_pos
        else:
            if self.num_dof < 6 or np.linalg.matrix_rank(J_num) < 6:
                print(
                    "⚠️ DOFs insuficientes para tarefa 6D. "
                    "Usando solução aproximada (DLS)."
                )
            weight_pos = 1.0
            weight_ang = 1.0
            J_task = np.vstack((weight_pos * J_pos, weight_ang * J_ang))
            v_command_6d = np.hstack(
                (weight_pos * v_command, weight_ang * (ang_vel_ff + orient_error * Kp_ik))
            )
            a_command_6d = np.hstack(
                (weight_pos * a_cartesian_target, weight_ang * a_ang_target)
            )
            drift_6d = np.hstack((weight_pos * drift_acc_pos, weight_ang * drift_acc_ang))
            J_task_pinv = J_task.T @ np.linalg.inv(
                J_task @ J_task.T + lambda_dls**2 * np.eye(6)
            )
            dq_task = J_task_pinv @ v_command_6d
            ddq_task = J_task_pinv @ (a_command_6d - drift_6d)
        
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
        target_rot=None,
        priority="position",
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

        if target_rot is not None:
            target_rot = np.array(target_rot, dtype=float)

        if target_rot is None:
            priority = "position"

        valid_priorities = {"position", "orientation", "balanced"}
        if priority not in valid_priorities:
            raise ValueError(
                f"Prioridade inválida ({priority}). Use uma de {valid_priorities}."
            )

        for i in range(max_iters):
            args = self._build_args(q_curr, np.zeros(self.num_dof))
            curr_pos = np.array(f_end(*args)).flatten()
            curr_rot = np.array(self.func_fk_rot(*args))
            error = target_pos - curr_pos
            orient_error = (
                self._orientation_error(target_rot, curr_rot)
                if target_rot is not None
                else np.zeros(3)
            )
            if priority == "position":
                error_vec = error
            elif priority == "orientation":
                error_vec = orient_error
            else:
                error_vec = np.hstack((error, orient_error))
            error_norm = np.linalg.norm(error_vec)

            if error_norm < tol:
                return q_curr, True, error_norm, i + 1

            J_num = np.array(self.func_J(*args))
            J_pos = J_num[:3, :]
            J_ang = J_num[3:6, :]
            if priority == "position":
                J_dls_pinv = J_pos.T @ np.linalg.inv(
                    J_pos @ J_pos.T + (lambda_dls**2) * np.eye(3)
                )
                dq = J_dls_pinv @ error
            elif priority == "orientation":
                orient_indices = list(range(self.num_dof))
                if self.has_vehicle_base and self.manipulator_dof_indices:
                    orient_indices = self.manipulator_dof_indices
                    J_ang_subset = J_ang[:, orient_indices]
                    if self.num_dof < 3 or np.linalg.matrix_rank(J_ang_subset) < 3:
                        print(
                            "⚠️ DOFs insuficientes na orientação do manipulador. "
                            "Usando todos os DOFs disponíveis."
                        )
                        orient_indices = list(range(self.num_dof))
                J_ang_masked = np.zeros_like(J_ang)
                J_ang_masked[:, orient_indices] = J_ang[:, orient_indices]
                J_ang_pinv = J_ang_masked.T @ np.linalg.inv(
                    J_ang_masked @ J_ang_masked.T + (lambda_dls**2) * np.eye(3)
                )
                dq_orient = J_ang_pinv @ orient_error
                null_projection = np.eye(self.num_dof) - J_ang_pinv @ J_ang_masked
                J_pos_null = J_pos @ null_projection
                J_pos_null_pinv = J_pos_null.T @ np.linalg.inv(
                    J_pos_null @ J_pos_null.T + (lambda_dls**2) * np.eye(3)
                )
                dq_pos = J_pos_null_pinv @ (error - J_pos @ dq_orient)
                dq = dq_orient + null_projection @ dq_pos
            else:
                J_task = np.vstack((J_pos, J_ang))
                error_task = np.hstack((error, orient_error))
                J_task_pinv = J_task.T @ np.linalg.inv(
                    J_task @ J_task.T + (lambda_dls**2) * np.eye(6)
                )
                dq = J_task_pinv @ error_task

            # Line search: reduz passo até melhorar o erro
            alpha = 1.0
            improved = False
            while alpha >= min_step:
                q_next = self._wrap_to_pi(q_curr + alpha * dq)
                args_next = self._build_args(q_next, np.zeros(self.num_dof))
                next_pos = np.array(f_end(*args_next)).flatten()
                next_error = target_pos - next_pos
                next_rot = np.array(self.func_fk_rot(*args_next))
                next_orient_error = (
                    self._orientation_error(target_rot, next_rot)
                    if target_rot is not None
                    else np.zeros(3)
                )
                if priority == "position":
                    next_error_vec = next_error
                elif priority == "orientation":
                    next_error_vec = next_orient_error
                else:
                    next_error_vec = np.hstack((next_error, next_orient_error))
                next_error_norm = np.linalg.norm(next_error_vec)
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

    def run(
        self,
        t_total,
        Pi_list,
        Pf_list,
        Kp_val,
        traj_mode="Line",
        traj_params=None,
        dt_physics=None,
        dt_visual=None,
        init_at_start=True,
        q_init=None,
        zeta=1.0,
        dq_limit=3.0,
        use_feedforward_vel=True,
        use_parallel=True,
        max_workers=None,
        orientation_preset=None,
        orientation_priority=None,
    ):
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

        priority = orientation_priority
        if self.num_dof < 6:
            if priority is None:
                priority = "balanced"
                print(
                    "ℹ️ DOFs < 6; prioridade não configurada, "
                    "usando fallback 'balanced'."
                )
            else:
                print(
                    f"ℹ️ DOFs < 6; usando prioridade configurada '{priority}'."
                )
        elif priority is None:
            priority = "position"
        
        # Inicialização (Com postura preferida se definida)
        q_home = np.copy(self.q_home) if hasattr(self, 'q_home') else np.zeros(self.num_dof)
        q = np.copy(q_home)
        dq = np.zeros(self.num_dof)

        if init_at_start:
            if q_init is None:
                q_init = np.copy(self.last_converged_q) if self.last_converged_q is not None else np.copy(q_home)
            else:
                q_init = np.array(q_init, dtype=float).copy()
            target_rot = self._derive_orientation_reference(
                orientation_preset,
                Pi,
                fallback_direction=(Pf - Pi),
            )
            init_priority = priority
            if target_rot is not None and init_priority == "position":
                init_priority = "balanced"
            q_init, converged, init_error, init_iters = self.solve_ik_initial(
                target_pos=Pi,
                q_init=q_init,
                max_iters=300,
                tol=1e-3,
                lambda_init=0.2,
                target_rot=target_rot,
                priority=init_priority,
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

        def _run_steps(executor):
            nonlocal current_time, q, dq
            for i in range(steps_visual):
                for _ in range(substeps):
                    current_time += dt_physics

                    # --- AQUI: CHAMADA DINÂMICA DO PLANEJADOR ---
                    P_ref, V_ref, A_ref = self.trajectory_planning(
                        current_time, t_total, Pi, Pf,
                        mode=traj_mode, params=traj_params
                    )
                    target_rot = self._derive_orientation_reference(
                        orientation_preset,
                        P_ref,
                        fallback_direction=V_ref,
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
                        target_rot=target_rot,
                        priority=priority,
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

        if use_parallel:
            worker_count = max_workers
            if worker_count is None:
                worker_count = max(1, min(3, os.cpu_count() or 1))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(self.sym_vars, self.expr_M, self.expr_C, self.expr_G),
            ) as executor:
                _run_steps(executor)
        else:
            _run_steps(None)

        return res_time, res_q, res_tau, anim_data
