import numpy as np
import sympy as sp

class RobotSimulator:
    def __init__(self, robot_math_instance, mode="Air"):
        self.bot = robot_math_instance
        self.mode = mode
        self.num_dof = len(self.bot.q)
        self.params_values = {} 

        print(f"[{mode}] Compilando equações simbólicas para binário (NumPy)...")
        
        # 1. Compila Dinâmica (M, C, G)
        self.sym_vars = self.bot.q + self.bot.dq + self.bot.params_list
        if hasattr(self.bot, 'rho'):
            self.sym_vars.append(self.bot.rho)
        
        # Cria funções rápidas (Lambdify)
        self.func_M = sp.lambdify(self.sym_vars, self.bot.M, modules='numpy')
        self.func_C = sp.lambdify(self.sym_vars, self.bot.C_total, modules='numpy')
        self.func_G = sp.lambdify(self.sym_vars, self.bot.G_vec, modules='numpy')
        
        # Compila Cinemática Direta (FK) - Pega posição da ponta
        self.func_FK = sp.lambdify(self.sym_vars, self.bot.frames[-1], modules='numpy')

        print("Compilação concluída!")

    def set_parameters(self, user_values_dict):
        """ Guarda valores numéricos (massas, comprimentos...) """
        self.params_values = user_values_dict

    # --- PLANEJAMENTO DE TRAJETÓRIA (Baseado em seus arquivos) ---
    def trajectory_planning(self, t, tf, type_traj, Pi, Pf):
        """ 
        Gera P, V, A para um instante t.
        Pi e Pf DEVEM ser numpy arrays.
        """
        if t >= tf:
            return Pf, np.zeros(3), np.zeros(3)
        
        if type_traj == "Line": 
            d = Pf - Pi
            modulo = np.linalg.norm(d)
            if modulo < 1e-12: return Pi, np.zeros(3), np.zeros(3)
            
            u = d / modulo
            
            # Polinômio Cúbico: s(t)
            # Condições: s(0)=0, s(tf)=distancia
            a2 = 3 * modulo / (tf**2)
            a3 = -2 * modulo / (tf**3)
            
            s = a2*(t**2) + a3*(t**3)
            sd = 2*a2*t + 3*a3*(t**2)
            sdd = 2*a2 + 6*a3*t
            
            P = Pi + s * u
            V = sd * u
            A = sdd * u
            return P, V, A
            
        return Pi, np.zeros(3), np.zeros(3) 

    def _build_args(self, q, dq):
        """ Monta a lista de argumentos na ordem correta para as funções lambdify """
        # A ordem é: q1..qn, dq1..dqn, params...
        p_vals = [self.params_values[str(p)] for p in self.bot.params_list]
        args = list(q) + list(dq) + p_vals
        if hasattr(self.bot, 'rho'):
            args.append(self.params_values['rho'])
        return args

    # --- LOOP PRINCIPAL DA SIMULAÇÃO ---
    def run(self, t_total, Pi_list, Pf_list, Kp_val, type_traj="Line"):
        dt = 0.05 # Passo de tempo (20Hz)
        steps = int(t_total / dt)
        time_span = np.linspace(0, t_total, steps)
        
        # --- CORREÇÃO DO ERRO ---
        # Converte listas para NumPy Arrays para poder fazer conta (Pf - Pi)
        Pi = np.array(Pi_list, dtype=float)
        Pf = np.array(Pf_list, dtype=float)
        
        # Arrays de Armazenamento
        res_q = np.zeros((steps, self.num_dof))
        res_tau = np.zeros((steps, self.num_dof))
        res_error = np.zeros((steps, self.num_dof))
        
        # Estado Inicial (Assume q=0, dq=0 ou faz inversa do Pi)
        q = np.zeros(self.num_dof) 
        dq = np.zeros(self.num_dof)
        
        # Ganhos do Controlador
        KP = Kp_val * np.eye(self.num_dof)
        KD = 2 * np.sqrt(Kp_val) * np.eye(self.num_dof) 
        
        print(f"Iniciando Simulação ({steps} passos)...")
        
        for i, t in enumerate(time_span):
            # 1. Planejamento
            P_ref, V_ref, A_ref = self.trajectory_planning(t, t_total, type_traj, Pi, Pf)
            
            # 2. Fake IK (Para teste)
            val = (t / t_total) * 0.5
            q_d = np.ones(self.num_dof) * val
            dq_d = np.ones(self.num_dof) * (0.5 / t_total)
            ddq_d = np.zeros(self.num_dof)

            # 3. Constroi Argumentos Numéricos
            args = self._build_args(q, dq)
            
            # --- BLINDAGEM CONTRA ERROS SIMBÓLICOS ---
            try:
                # Tenta calcular e força conversão para float
                # Se sobrar símbolo aqui (dtype='O'), vai dar erro no astype
                M = np.array(self.func_M(*args)).astype(np.float64)
                C = np.array(self.func_C(*args)).flatten().astype(np.float64)
                G = np.array(self.func_G(*args)).flatten().astype(np.float64)
            except Exception as e:
                print(f"❌ ERRO MATEMÁTICO NO PASSO {i}:")
                print("Alguma variável simbólica não foi substituída por número.")
                print("Verifique se todos os params (vol, rho, m...) estão no params_list.")
                # Debug: imprime o que o lambdify gerou
                raw_M = np.array(self.func_M(*args))
                print("Matriz M contém:", raw_M.dtype)
                print(raw_M) 
                raise e # Para a simulação

            # Erros
            e = q_d - q
            e_dot = dq_d - dq
            
            # Controle
            u = ddq_d + (KD @ e_dot) + (KP @ e)
            tau = M @ u + C + G
            
            # 4. Planta
            rhs = tau - C - G
            
            # Agora M e rhs são float garantidos, o solve não vai falhar com casting error
            ddq = np.linalg.solve(M, rhs)
            
            # 5. Integração
            q = q + dq * dt
            dq = dq + ddq * dt
            
            res_q[i, :] = e
            res_tau[i, :] = tau
            
        return time_span, res_q, res_tau