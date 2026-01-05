import sympy as sp
import numpy as np

class SimulationEngine:
    def __init__(self, robot_engine):
        self.bot = robot_engine
        self.fast_funcs = {}
        
    def compile_symbols(self):
        """Transforma o SymPy em NumPy para rodar rápido"""
        # Define valores padrão para simulação (Massa 1kg, Elo 1m) para teste
        numeric_params = {}
        for sym in self.bot.params_list:
            numeric_params[str(sym)] = 1.0
        
        # Gravidade e densidade
        numeric_params['g'] = 9.81
        numeric_params['rho'] = 1000.0
        
        # Cria lista de substituição
        subs = []
        for sym in self.bot.params_list:
            if str(sym) in numeric_params:
                subs.append((sym, numeric_params[str(sym)]))
        subs.append((self.bot.g, 9.81))
        if hasattr(self.bot, 'rho'): subs.append((self.bot.rho, 1000.0))
        
        # Compila M, G, J, FK
        self.fast_funcs['M'] = sp.lambdify([self.bot.q], self.bot.M.subs(subs), 'numpy')
        self.fast_funcs['G'] = sp.lambdify([self.bot.q], self.bot.G_vec.subs(subs), 'numpy')
        self.fast_funcs['J'] = sp.lambdify([self.bot.q], self.bot.Jacobian.subs(subs), 'numpy')
        self.fast_funcs['FK'] = sp.lambdify([self.bot.q], self.bot.frames[-1].subs(subs), 'numpy')
        
        # Compila Elos para Animação
        self.fast_funcs['Links'] = []
        for f in self.bot.frames:
            self.fast_funcs['Links'].append(sp.lambdify([self.bot.q], f.subs(subs), 'numpy'))

    def run_simulation(self, target_xyz, duration=4.0):
        # Configuração da Simulação
        dt = 0.005 # 5ms
        steps = int(duration / dt)
        
        n_juntas = len(self.bot.q)
        q = np.zeros(n_juntas)
        dq = np.zeros(n_juntas)
        integral_error = np.zeros(n_juntas)
        
        t_hist, q_hist, q_des_hist = [], [], []
        torque_hist, error_joint_hist = [], []
        
        target_pos = np.array(target_xyz)
        
        for i in range(steps):
            t = i * dt
            
            # --- 1. Cinemática Inversa (Drift Correction) ---
            T_curr = self.fast_funcs['FK'](q)
            curr_pos = T_curr[0:3, 3].flatten()
            err_cart = target_pos - curr_pos
            
            J = self.fast_funcs['J'](q)
            J_lin = J[0:3, :]
            
            lamb = 0.05
            J_pinv = J_lin.T @ np.linalg.inv(J_lin @ J_lin.T + lamb**2 * np.eye(3))
            
            dq_ref = J_pinv @ (2.0 * err_cart)
            q_des = q + dq_ref * dt
            
            # --- 2. Controle PID ---
            Kp = 80.0
            Ki = 40.0 
            Kd = 10.0
            
            e_joint = q_des - q
            de_joint = dq_ref - dq
            integral_error += e_joint * dt
            
            M_num = self.fast_funcs['M'](q)
            G_num = self.fast_funcs['G'](q).flatten()
            
            # Tau = M(q)*u + G(q)
            u_pid = Kp * e_joint + Ki * integral_error + Kd * de_joint
            tau = M_num @ u_pid + G_num
            
            # --- 3. Planta ---
            # Aceleração = M_inv * (Tau - G - Atrito)
            ddq = np.linalg.solve(M_num, tau - G_num - 0.5*dq)
            
            dq += ddq * dt
            q += dq * dt
            
            t_hist.append(t)
            q_hist.append(q.copy())
            q_des_hist.append(q_des.copy())
            torque_hist.append(tau)
            error_joint_hist.append(e_joint)
            
        return t_hist, q_hist, q_des_hist, torque_hist, error_joint_hist