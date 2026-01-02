import customtkinter as ctk
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.octave import octave_code
import os
import threading
import sys

# ==============================================================================
# 1. ENGINE MATEMÁTICA (SEU CÓDIGO COMPLETO + LOGGING)
# ==============================================================================
class RobotMathEngine:
    def __init__(self, joint_config, link_vectors_mask, logger_callback=None):
        self.joint_config = joint_config
        # Converte para Matrix para garantir precisão
        self.link_vectors_mask = [sp.Matrix(v) for v in link_vectors_mask]
        self.log = logger_callback if logger_callback else print

        self.t = sp.symbols('t')
        self.g = sp.symbols('g')

        self.q = []
        self.dq = []
        self.params_list = []

        self.frames = []
        self.rotation_matrices = []
        self.com_positions_global = []
        self.angular_velocities = []
        self.masses = []

        self.M = None
        self.G_vec = None
        self.C_total = None
        self.Jacobian = None

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
        
        # Se for muito grande, avisa ou pula
        if len(self.q) > 8:
            self.log("⚠️ AVISO: Sistema com muitos graus de liberdade.")
            self.log("O cálculo de Coriolis pode demorar alguns minutos...")
        
        self.step_3_coriolis_combined()
        return self.step_4_prepare_export()

    def step_1_kinematics(self):
        self.log("1. Calculando Cinemática (Rotações e CMs)...")
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
            # Assume CM no centro do elo deslocado para simplificar input da GUI, 
            # mas mantendo variaveis simbolicas se quiser expandir depois
            cx, cy, cz = sp.symbols(f'cx{i+1} cy{i+1} cz{i+1}')
            
            self.masses.append(m)
            self.params_list.extend([m, L])

            type_char, axis_char = j_type[0], j_type[1].lower()
            axis_vec_local = self._get_axis_vector(axis_char)

            # Rotação
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

            # Geometria da Junta
            T_joint = sp.eye(4)
            T_joint[0:3, 0:3] = R_j
            T_joint[0:3, 3] = P_j
            T_at_joint_start = T_acc * T_joint

            # Elo (Geometria Fixa) - Multiplica pelo escalar L
            P_link_vec = link_vec * L
            T_link = sp.eye(4)
            T_link[0:3, 3] = P_link_vec
            T_acc = T_at_joint_start * T_link
            self.frames.append(T_acc)

            # CM Global (Aproximação para o centro do Link definido)
            # Pega o ponto inicial e final do elo e tira a média
            p_start = T_at_joint_start[0:3, 3]
            p_end = T_acc[0:3, 3]
            # Usa symbolicamente cx, cy, cz se eles fossem definidos localmente, 
            # mas aqui vamos usar a geometria do link para posicionar o CM
            self.com_positions_global.append((p_start + p_end)/2)


    def step_2_jacobian_M_G(self):
        self.log("2. Dinâmica M e G (Com Tensores de Inércia)...")
        n = len(self.q)
        self.M = sp.zeros(n, n)
        V_tot = 0

        for i in range(n):
            m = self.masses[i]
            pos_cm = self.com_positions_global[i]
            R_global = self.rotation_matrices[i]

            J_v = pos_cm.jacobian(self.q)
            J_w = self.angular_velocities[i].jacobian(self.dq)

            # Tensor de Inércia Simbólico
            Ixx, Iyy, Izz = sp.symbols(f'Ixx{i+1} Iyy{i+1} Izz{i+1}')
            self.params_list.extend([Ixx, Iyy, Izz])
            I_local = sp.Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
            
            # Rotaciona Tensor
            I_global = R_global * I_local * R_global.T

            # Energia Cinética (Translacional + Rotacional)
            self.M += m * J_v.T * J_v + J_w.T * I_global * J_w
            
            # Energia Potencial
            V_tot += m * self.g * pos_cm[2]

        self.G_vec = sp.Matrix([V_tot]).jacobian(self.q).T
        
        # Jacobiano da Ponta (Linear e Angular)
        J_lin = self.frames[-1][0:3, 3].jacobian(self.q)
        J_ang = self.angular_velocities[-1].jacobian(self.dq)
        self.Jacobian = J_lin.col_join(J_ang)


    def step_3_coriolis_combined(self):
        self.log("3. Calculando Coriolis (Isso pode demorar)...")
        n = len(self.q)
        self.C_total = sp.zeros(n, 1)

        self.log(f"   -> Derivando Matriz M ({n}x{n})...")
        dM_dq = []
        for i, qk in enumerate(self.q):
            dM_dq.append(self.M.diff(qk))

        self.log(f"   -> Montando Christoffel e Vetor C...")
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
            self.log(f"      Linha {i+1} calculada.")

    def step_4_prepare_export(self):
        self.log("4. Otimizando equações para exportação...")
        
        # Mapeamento para nomes bonitos (q1, q2...)
        mapa_subs = {}
        for i in range(len(self.q)):
            mapa_subs[self.q[i]] = sp.Symbol(f'q{i+1}')
            mapa_subs[self.dq[i]] = sp.Symbol(f'dq{i+1}')

        # Cinemática de Velocidade da ponta (V = J * dq)
        dq_vec = sp.Matrix(self.dq)
        if self.Jacobian is not None:
             V_cartesian = self.Jacobian * dq_vec
        else:
             V_cartesian = None

        return {
            "M": self.M,
            "G": self.G_vec,
            "C": self.C_total,
            "J": self.Jacobian,
            "FK_Pos": self.frames[-1],
            "FK_Vel": V_cartesian,
            "Subs": mapa_subs
        }

# ==============================================================================
# 2. INTERFACE GRÁFICA (CUSTOMTKINTER)
# ==============================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Hephaestus v1.0") 
        self.geometry("1100x750")

        # --- ADICIONE ISSO AQUI: ---
        # O try/except evita que o programa trave se não achar o ícone
        try:
            self.iconbitmap("hephaestus.ico")
        except:
            pass # Se não achar o ícone, usa o padrão sem dar erro
        
        # ... (resto do código continua igual)
        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="CONFIGURAÇÃO", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.pack(pady=20)

        self.btn_add = ctk.CTkButton(self.sidebar, text="+ Adicionar Junta", command=self.add_joint)
        self.btn_add.pack(pady=10, padx=20)
        
        self.btn_remove = ctk.CTkButton(self.sidebar, text="- Remover Última", command=self.remove_joint, fg_color="firebrick")
        self.btn_remove.pack(pady=10, padx=20)

        self.btn_about = ctk.CTkButton(self.sidebar, text="Sobre / Créditos", command=self.open_about, fg_color="gray", hover_color="gray30")
        self.btn_about.pack(pady=10, padx=20, side="bottom")

        self.btn_calc = ctk.CTkButton(self.sidebar, text="GERAR MODELO 🚀", command=self.start_processing, fg_color="green", height=50, font=ctk.CTkFont(weight="bold"))
        self.btn_calc.pack(pady=40, padx=20, side="bottom")

        # --- Área Principal ---
        self.main_area = ctk.CTkFrame(self)
        self.main_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.lbl_title = ctk.CTkLabel(self.main_area, text="Cadeia Cinemática e Excentricidades", font=ctk.CTkFont(size=18))
        self.lbl_title.pack(pady=10)

        # ScrollFrame Juntas
        self.scroll_frame = ctk.CTkScrollableFrame(self.main_area, label_text="Juntas Ativas")
        self.scroll_frame.pack(expand=True, fill="both", padx=10, pady=5)

        self.joint_rows = []
        self.add_joint() # Começa com 1 junta

        # Log
        self.lbl_log = ctk.CTkLabel(self.main_area, text="Log de Processamento:", anchor="w")
        self.lbl_log.pack(fill="x", padx=10, pady=(10,0))
        
        self.status_bar = ctk.CTkTextbox(self.main_area, height=150, font=("Consolas", 12))
        self.status_bar.pack(fill="x", padx=10, pady=10)
        self.log("Sistema pronto. Adicione juntas e configure os vetores dos elos.")


    def open_about(self):
        # Cria uma janela nova que fica na frente da principal (Toplevel)
        about = ctk.CTkToplevel(self)
        about.title("Sobre")
        about.geometry("500x450")
        about.resizable(False, False)
        
        # Garante que a janela fique no topo e em foco
        about.attributes("-topmost", True)
        
        # Conteúdo da janela
        ctk.CTkLabel(about, text="Hephaestus v1.0", font=("Arial", 24, "bold"), text_color="#3B8ED0").pack(pady=(30, 10))
        ctk.CTkLabel(about, text="Modelagem Simbólica para Robótica (UVMS)", font=("Arial", 14)).pack(pady=5)
        
        # SEUS DADOS AQUI
        texto = (
            "\nDesenvolvido por:\n"
            "Lucas da Silva Santos\n\n"
            "Orientadores:\n"
            "Luciano Santos Constantin Raptopoulos\n"
            "Josiel Alves Gouvêa\n\n"
            "Instituto:\n"
            "Centro Federal de Educação Tecnológica Celso Suckow da Fonseca (CEFET/RJ)\n\n"
            "Curso\n"
            "Engenharia Industrial de Controle e Automação\n\n"
            "Ano: 2026"
        )
        ctk.CTkLabel(about, text=texto, justify="center").pack(pady=10)
        
        ctk.CTkButton(about, text="Fechar", command=about.destroy, width=100).pack(pady=20)

    def log(self, msg):
        self.status_bar.insert("end", str(msg) + "\n")
        self.status_bar.see("end")

    def add_joint(self):
        idx = len(self.joint_rows) + 1
        if idx > 12: return 

        row = ctk.CTkFrame(self.scroll_frame)
        row.pack(fill="x", pady=5)

        # UI Elementos
        ctk.CTkLabel(row, text=f"Junta {idx}:", width=50).pack(side="left", padx=5)
        
        dd = ctk.CTkOptionMenu(row, values=["Rz", "Ry", "Rx", "Dz", "Dy", "Dx"], width=70)
        dd.pack(side="left", padx=5)

        ctk.CTkLabel(row, text="Elo(L):").pack(side="left", padx=10)
        
        cx = ctk.CTkCheckBox(row, text="X", width=40)
        cx.pack(side="left", padx=2)
        cy = ctk.CTkCheckBox(row, text="Y", width=40)
        cy.pack(side="left", padx=2)
        cz = ctk.CTkCheckBox(row, text="Z", width=40)
        cz.pack(side="left", padx=2)

        # Default Z
        if idx == 1: cz.select()

        self.joint_rows.append({"frame": row, "dd": dd, "cx": cx, "cy": cy, "cz": cz})

    def remove_joint(self):
        if len(self.joint_rows) > 0:
            row = self.joint_rows.pop()
            row["frame"].destroy()

    def start_processing(self):
        self.btn_calc.configure(state="disabled", text="Processando...")
        self.status_bar.delete("0.0", "end")
        threading.Thread(target=self.run_logic).start()

    def run_logic(self):
        try:
            # 1. Coleta Inputs
            j_types = []
            l_vecs = []
            for item in self.joint_rows:
                j_types.append(item["dd"].get())
                vx = 1 if item["cx"].get() else 0
                vy = 1 if item["cy"].get() else 0
                vz = 1 if item["cz"].get() else 0
                l_vecs.append([vx, vy, vz])

            self.log(f"Iniciando cálculo para {len(j_types)} DOFs...")

            # 2. Instancia Engine (Passando self.log como callback)
            bot = RobotMathEngine(j_types, l_vecs, logger_callback=self.log)
            results = bot.run_full_process()

            # 3. Exporta Arquivos
            folder = "Resultados_Robo"
            if not os.path.exists(folder): os.makedirs(folder)

            self.log(f"Salvando arquivos na pasta '{folder}'...")
            mapa = results["Subs"]

            def save(name, expr):
                if expr is None: return
                with open(f"{folder}/{name}.txt", "w") as f:
                    f.write(f"% --- {name} ---\n")
                    try:
                        # Tenta usar octave_code com substituição
                        if hasattr(expr, 'subs'):
                            f.write(octave_code(expr.subs(mapa)))
                        else:
                            f.write(str(expr))
                    except:
                        f.write(str(expr))

            save("Matriz_M", results["M"])
            save("Vetor_G", results["G"])
            save("Vetor_C", results["C"])
            save("Cinematica_Posicao", results["FK_Pos"])
            save("Cinematica_Velocidade", results["FK_Vel"])
            
            if results["J"] is not None:
                save("Jacobiano_Linear", results["J"][0:3, :])
                save("Jacobiano_Angular", results["J"][3:6, :])

            self.log("\n✅ SUCESSO! Todos os arquivos foram gerados.")
            self.log(f"Verifique a pasta: {os.path.abspath(folder)}")

        except Exception as e:
            self.log(f"\n❌ ERRO: {str(e)}")
        
        finally:
            self.btn_calc.configure(state="normal", text="GERAR MODELO 🚀")

if __name__ == "__main__":
    app = App()
    app.mainloop()