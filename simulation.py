import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline

class RobotSimulator:
    def __init__(self, robot_math_instance, mode="Air"):
        self.bot = robot_math_instance
        self.mode = mode
        self.num_dof = len(self.bot.q)
        self.params_values = {} 

        print(f"[{mode}] Compiling symbolic equations to NumPy functions...")
        
        # 1. Compile Dynamics (M, C, G)
        # Gather all symbols: q, dq, params (masses, lengths, inertias, etc.)
        self.sym_vars = self.bot.q + self.bot.dq + self.bot.params_list
        if hasattr(self.bot, 'rho'):
            self.sym_vars.append(self.bot.rho)
        
        # Create fast numeric functions from SymPy expressions
        self.func_M = sp.lambdify(self.sym_vars, self.bot.M, modules='numpy')
        self.func_C = sp.lambdify(self.sym_vars, self.bot.C_total, modules='numpy')
        self.func_G = sp.lambdify(self.sym_vars, self.bot.G_vec, modules='numpy')
        
        # Compile Forward Kinematics (for feedback)
        # Assumes the last frame is the end-effector
        self.func_FK = sp.lambdify(self.sym_vars, self.bot.frames[-1], modules='numpy')

        print("Compilation complete!")

    def set_parameters(self, user_values_dict):
        """ Stores numerical values for mass, lengths, etc. """
        self.params_values = user_values_dict

    # --- TRAJECTORY PLANNING (Based on planejamentos.txt) ---
    def trajectory_planning(self, t, tf, type_traj, Pi, Pf):
        """ 
        Generates P, V, A for a specific time t 
        Matches logic from 'planejamentos.txt' (Cubic) 
        """
        if t >= tf:
            return Pf, np.zeros(3), np.zeros(3)
        
        if type_traj == "Line": # FCubica logic [cite: 280]
            d = Pf - Pi
            modulo = np.linalg.norm(d)
            if modulo < 1e-12: return Pi, np.zeros(3), np.zeros(3)
            
            u = d / modulo
            # Cubic interpolation for scalar s(t)
            # Coefficients for s(t) = a0 + a1*t + a2*t^2 + a3*t^3
            # Boundary conditions: s(0)=0, s(tf)=modulo, v(0)=0, v(tf)=0
            a0 = 0
            a1 = 0
            a2 = 3 * modulo / (tf**2)
            a3 = -2 * modulo / (tf**3)
            
            s = a0 + a1*t + a2*t**2 + a3*t**3
            sd = a1 + 2*a2*t + 3*a3*t**2
            sdd = 2*a2 + 6*a3*t
            
            P = Pi + s * u
            V = sd * u
            A = sdd * u
            return P, V, A
            
        elif type_traj == "Circle": # PlanCircular logic [cite: 290]
            # (Simplified implementation of the circular logic provided)
            # For brevity, implementing a placeholder or full logic if needed.
            # ... Circular logic implementation ...
            return Pi, np.zeros(3), np.zeros(3) 

    # --- INVERSE KINEMATICS (Based on inversa.txt) ---
    def inverse_kinematics_numerical(self, P_des, V_des, A_des, q_curr, dt):
        """
        Calculates Qd, dQd, d2Qd numerically using the Jacobian.
        This replaces the analytical 'inversa.txt'  for genericity across N-DOF robots.
        """
        # Numerical Inverse Kinematics (CLIK algorithm)
        # 1. Forward Kinematics at current q
        args_num = self._build_args(q_curr, np.zeros(self.num_dof))
        T_curr = np.array(self.func_FK(*args_num))
        P_curr = T_curr[:3, 3] # Position X,Y,Z
        
        # 2. Numerical Jacobian
        # We need to calculate J numerically or use the symbolic one if exported.
        # Since we use lambdify, we need to create a function for J in __init__ if not present.
        # For now, let's assume a simple Jacobian calculation or finite difference if J is missing.
        # (Recommendation: Add func_J to __init__)
        
        # Placeholder for CLIK logic:
        # J = ...
        # dQ = pinv(J) * (V_des + Kp * (P_des - P_curr))
        # Q = q_curr + dQ * dt
        
        # For this example, let's assume we return target states directly
        # In a real implementation, you'd use your symbolic Jacobian here.
        return q_curr, np.zeros(self.num_dof), np.zeros(self.num_dof)

    def _build_args(self, q, dq):
        """ Helper to construct the argument list for lambdified functions """
        # args order: q... dq... params...
        p_vals = [self.params_values[str(p)] for p in self.bot.params_list]
        args = list(q) + list(dq) + p_vals
        if hasattr(self.bot, 'rho'):
            args.append(self.params_values['rho'])
        return args

    # --- MAIN SIMULATION LOOP ---
    def run(self, t_total, Pi, Pf, Kp_val, type_traj="Line"):
        dt = 0.01
        steps = int(t_total / dt)
        time_span = np.linspace(0, t_total, steps)
        
        # Storage
        res_q = np.zeros((steps, self.num_dof))
        res_tau = np.zeros((steps, self.num_dof))
        res_error = np.zeros((steps, self.num_dof))
        
        # Initial State
        q = np.zeros(self.num_dof) # Assume starting at 0 or solve IK for Pi
        dq = np.zeros(self.num_dof)
        
        # Control Gains 
        KP = Kp_val * np.eye(self.num_dof)
        KD = 2 * np.sqrt(Kp_val) * np.eye(self.num_dof) # Critical damping
        
        print("Starting Simulation...")
        
        for i, t in enumerate(time_span):
            # 1. Trajectory Planning 
            P_ref, V_ref, A_ref = self.trajectory_planning(t, t_total, type_traj, Pi, Pf)
            
            # 2. Inverse Kinematics 
            # (Here we ideally use the Numerical Inverse Kinematics to get joint targets)
            # For now, let's act as if q_d is calculated. 
            # q_d, dq_d, ddq_d = self.inverse_kinematics_numerical(P_ref, V_ref, A_ref, q, dt)
            
            # Temporary bypass for testing dynamics: 
            # Let's say we want to hold q=0.5
            q_d = np.ones(self.num_dof) * 0.5 * (t/t_total)
            dq_d = np.ones(self.num_dof) * 0.5 / t_total
            ddq_d = np.zeros(self.num_dof)

            # 3. Controller (PID + Feedforward) 
            # Tc = M(ddqd + Kd*e_dot + Kp*e) + C + G
            args = self._build_args(q, dq)
            M = np.array(self.func_M(*args))
            C = np.array(self.func_C(*args)).flatten()
            G = np.array(self.func_G(*args)).flatten()
            
            e = q_d - q
            e_dot = dq_d - dq
            
            # Control Law [cite: 269]
            # Tc = M * (ddq_d + KD @ e_dot + KP @ e) + C + G
            # Note: H in your text usually refers to Coriolis/Centripetal (C) or friction.
            # Assuming H is included in C or neglected for now.
            
            term_pid = ddq_d + (KD @ e_dot) + (KP @ e)
            tau = M @ term_pid + C + G
            
            # 4. Plant Dynamics (Forward Dynamics) 
            # ddq = M \ (tau - C - G)
            rhs = tau - C - G
            ddq = np.linalg.solve(M, rhs)
            
            # 5. Integration (Euler)
            q = q + dq * dt
            dq = dq + ddq * dt
            
            # Store
            res_q[i, :] = e # Storing error for plotting
            res_tau[i, :] = tau
            
        return time_span, res_q, res_tau