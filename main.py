import customtkinter as ctk
import json
import threading
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- IMPORTAÇÕES DOS SEUS MÓDULOS ---
# Certifique-se que engine.py e simulator.py estão na mesma pasta
from engine import RobotMathEngine, RobotMathHydro
from simulator import RobotSimulator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hephaestus v4.0 - Integrated Environment")
        self.geometry("1200x850")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variáveis de Estado
        self.active_bot = None       # Instância da Modelagem Matemática
        self.active_sim = None       # Instância do Simulador Numérico
        self.joint_rows = []         # Lista para guardar os widgets das juntas
        
        # Configuração do Grid Principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- ABAS ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.tab_model = self.tabview.add("Modelagem")
        self.tab_sim = self.tabview.add("Simulação")
        
        # Constrói as telas
        self.setup_modeling_tab()
        self.setup_simulation_tab()
        
        # Bloqueia simulação inicialmente
        self.toggle_sim_tab(False)

    # ==========================================================================
    # ABA 1: MODELAGEM (RESTAURADA)
    # ==========================================================================
    def setup_modeling_tab(self):
        # Layout: Esquerda (Configuração) | Direita (Log/Status)
        self.tab_model.grid_columnconfigure(0, weight=3) # Configuração maior
        self.tab_model.grid_columnconfigure(1, weight=2) # Log menor
        self.tab_model.grid_rowconfigure(0, weight=1)

        # --- COLUNA DA ESQUERDA: CONFIGURAÇÃO ---
        left_frame = ctk.CTkFrame(self.tab_model)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # 1. Seletor de Modo (Topo)
        mode_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        mode_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(mode_frame, text="Ambiente:", font=("Arial", 12, "bold")).pack(side="left", padx=5)
        self.mode_var = ctk.StringVar(value="Ar (Seco)")
        self.mode_switch = ctk.CTkSegmentedButton(mode_frame, values=["Ar (Seco)", "Água (UVMS)"], 
                                                  variable=self.mode_var, command=self.update_mode_color)
        self.mode_switch.pack(side="left", padx=10)
        self.update_mode_color("Ar (Seco)") # Cor inicial

        # 2. Lista de Juntas (Scrollable - Onde estava faltando!)
        self.scroll_joints = ctk.CTkScrollableFrame(left_frame, label_text="Cadeia Cinemática")
        self.scroll_joints.pack(expand=True, fill="both", padx=10, pady=5)
        
        # Adiciona a primeira junta por padrão
        self.add_joint()

        # 3. Botões de Controle de Juntas
        ctrl_joints_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        ctrl_joints_frame.pack(fill="x", padx=10, pady=5)
        
        btn_add = ctk.CTkButton(ctrl_joints_frame, text="+ Adicionar Junta", command=self.add_joint)
        btn_add.pack(side="left", expand=True, padx=2)
        
        btn_rem = ctk.CTkButton(ctrl_joints_frame, text="- Remover Última", command=self.remove_joint, fg_color="firebrick")
        btn_rem.pack(side="left", expand=True, padx=2)

        # 4. Ações Principais (Gerar / JSON)
        action_frame = ctk.CTkFrame(left_frame)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        self.btn_calc = ctk.CTkButton(action_frame, text="GERAR MODELO 🚀", command=self.run_modeling, 
                                      height=40, font=ctk.CTkFont(weight="bold"), fg_color="green")
        self.btn_calc.pack(fill="x", padx=10, pady=(10, 5))
        
        btn_save = ctk.CTkButton(action_frame, text="Salvar Projeto (.json)", command=self.save_json)
        btn_save.pack(fill="x", padx=10, pady=2)
        
        btn_load = ctk.CTkButton(action_frame, text="Carregar Projeto (.json)", command=self.load_json)
        btn_load.pack(fill="x", padx=10, pady=(2, 10))

        # --- COLUNA DA DIREITA: LOG ---
        right_frame = ctk.CTkFrame(self.tab_model)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(right_frame, text="Log de Processamento").pack(pady=5)
        self.status_bar = ctk.CTkTextbox(right_frame, font=("Consolas", 12))
        self.status_bar.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.log("Sistema inicializado. Configure o robô à esquerda.")

    # --- LÓGICA DE INTERFACE (Juntas) ---
    def add_joint(self):
        idx = len(self.joint_rows) + 1
        if idx > 12: return 

        row = ctk.CTkFrame(self.scroll_joints)
        row.pack(fill="x", pady=2)

        # Label
        ctk.CTkLabel(row, text=f"Junta {idx}:", width=50).pack(side="left", padx=5)
        
        # Tipo
        dd = ctk.CTkOptionMenu(row, values=["Rz", "Ry", "Rx", "Dz", "Dy", "Dx"], width=70)
        dd.pack(side="left", padx=5)

        # Elos
        ctk.CTkLabel(row, text="Elo(L):").pack(side="left", padx=5)
        cx = ctk.CTkCheckBox(row, text="X", width=40)
        cx.pack(side="left", padx=2)
        cy = ctk.CTkCheckBox(row, text="Y", width=40)
        cy.pack(side="left", padx=2)
        cz = ctk.CTkCheckBox(row, text="Z", width=40)
        cz.pack(side="left", padx=2)

        # Default Z para o primeiro
        if idx == 1: cz.select()

        self.joint_rows.append({"frame": row, "dd": dd, "cx": cx, "cy": cy, "cz": cz})

    def remove_joint(self):
        if len(self.joint_rows) > 1: # Mantém pelo menos 1
            row = self.joint_rows.pop()
            row["frame"].destroy()

    def update_mode_color(self, value):
        if value == "Água (UVMS)":
            self.mode_switch.configure(selected_color="#1E90FF", selected_hover_color="#104E8B")
        else:
            self.mode_switch.configure(selected_color="#2E8B57", selected_hover_color="#228B22")

    def log(self, msg):
        self.status_bar.insert("end", str(msg) + "\n")
        self.status_bar.see("end")
        print(msg) # Também imprime no terminal para debug

    # ==========================================================================
    # LÓGICA DE GERAÇÃO (CORRIGIDA)
    # ==========================================================================
    def run_modeling(self):
        self.btn_calc.configure(state="disabled", text="Calculando...")
        # Roda em thread para não travar a GUI
        threading.Thread(target=self._run_modeling_thread, daemon=True).start()

    def _run_modeling_thread(self):
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

            modo = self.mode_var.get()
            self.log(f"--- Iniciando Modelagem ({modo}) ---")
            self.log(f"Graus de Liberdade: {len(j_types)}")

            # 2. Instancia Engine
            if modo == "Água (UVMS)":
                self.active_bot = RobotMathHydro(j_types, l_vecs)
            else:
                self.active_bot = RobotMathEngine(j_types, l_vecs)

            # 3. Calcula Simbólico
            results = self.active_bot.run_full_process()
            
            # 4. Salva Arquivos txt (Opcional, mas bom para backup)
            folder = "Output_Files"
            if not os.path.exists(folder): os.makedirs(folder)
            self.log(f"Salvando equações em .txt na pasta '{folder}'...")
            # (Aqui você pode adicionar a lógica de salvar txt se quiser, igual ao anterior)

            # 5. Prepara Simulação
            self.log("Compilando equações para o Simulador Numérico...")
            sim_mode = "Hydro" if modo == "Água (UVMS)" else "Air"
            
            # AQUI ESTAVA O ERRO: active_bot agora existe!
            self.active_sim = RobotSimulator(self.active_bot, mode=sim_mode)
            
            # 6. Atualiza GUI da Simulação
            # Precisa rodar na main thread do Tkinter
            self.after(0, self.finish_modeling_success)

        except Exception as e:
            self.log(f"ERRO CRÍTICO: {str(e)}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self.btn_calc.configure(state="normal", text="GERAR MODELO 🚀"))

    def finish_modeling_success(self):
        self.generate_sim_inputs()
        self.toggle_sim_tab(True)
        self.tabview.set("Simulação")
        self.log("✅ Modelagem e Compilação concluídas com sucesso!")
        self.btn_calc.configure(state="normal", text="GERAR MODELO 🚀")

    # ==========================================================================
    # ABA 2: SIMULAÇÃO
    # ==========================================================================
    def setup_simulation_tab(self):
        self.tab_sim.grid_columnconfigure(0, weight=1) # Params
        self.tab_sim.grid_columnconfigure(1, weight=3) # Gráficos
        self.tab_sim.grid_rowconfigure(0, weight=1)

        # --- Esquerda: Parâmetros ---
        self.sim_left = ctk.CTkScrollableFrame(self.tab_sim, label_text="Parâmetros de Simulação")
        self.sim_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Trajetória
        ctk.CTkLabel(self.sim_left, text="Posição Inicial (x, y, z):").pack(anchor="w")
        self.entry_start = ctk.CTkEntry(self.sim_left)
        self.entry_start.insert(0, "0.5, 0.0, 0.0")
        self.entry_start.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(self.sim_left, text="Posição Final (x, y, z):").pack(anchor="w")
        self.entry_end = ctk.CTkEntry(self.sim_left)
        self.entry_end.insert(0, "0.8, 0.2, 0.5")
        self.entry_end.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.sim_left, text="Tempo Total (s):").pack(anchor="w")
        self.entry_time = ctk.CTkEntry(self.sim_left)
        self.entry_time.insert(0, "5.0")
        self.entry_time.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.sim_left, text="Ganho Kp (Controle):").pack(anchor="w")
        self.entry_kp = ctk.CTkEntry(self.sim_left)
        self.entry_kp.insert(0, "100.0")
        self.entry_kp.pack(fill="x", pady=(0, 20))

        # Container dinâmico para Params Físicos (Massas, Inércias...)
        ctk.CTkLabel(self.sim_left, text="--- Constantes do Robô ---", font=("Arial", 12, "bold")).pack(pady=5)
        self.params_container = ctk.CTkFrame(self.sim_left, fg_color="transparent")
        self.params_container.pack(fill="both", expand=True)
        self.dynamic_entries = {}

        self.btn_run_sim = ctk.CTkButton(self.sim_left, text="RODAR SIMULAÇÃO ▶", fg_color="red", command=self.run_simulation_logic)
        self.btn_run_sim.pack(pady=20, side="bottom", fill="x")

        # --- Direita: Gráficos ---
        self.sim_right = ctk.CTkFrame(self.tab_sim)
        self.sim_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Placeholder para o Matplotlib
        self.plot_frame = ctk.CTkFrame(self.sim_right, fg_color="white") # Cor de fundo do plot
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def generate_sim_inputs(self):
        # Limpa anteriores
        for widget in self.params_container.winfo_children():
            widget.destroy()
        self.dynamic_entries = {}

        if not self.active_sim: return

        # Gera inputs para cada símbolo necessário (m1, L1, Ixx...)
        # active_sim.sym_vars contém a lista de símbolos
        # Vamos filtrar para não mostrar q e dq, apenas parâmetros
        
        ignore_list = set(self.active_bot.q + self.active_bot.dq)
        
        for sym in self.active_sim.sym_vars:
            if sym in ignore_list: continue
            if str(sym) == 't' or str(sym) == 'g': continue # Ignora tempo e g (já fixo)

            row = ctk.CTkFrame(self.params_container)
            row.pack(fill="x", pady=2)
            
            lbl = ctk.CTkLabel(row, text=str(sym), width=80, anchor="w")
            lbl.pack(side="left")
            
            entry = ctk.CTkEntry(row)
            entry.pack(side="right", expand=True, fill="x")
            
            # Valores padrão inteligentes
            val_padrao = "0.1"
            if "rho" in str(sym): val_padrao = "1000"
            if "m" in str(sym): val_padrao = "2.0"
            if "vol" in str(sym): val_padrao = "0.005"
            
            entry.insert(0, val_padrao)
            self.dynamic_entries[str(sym)] = entry

    def toggle_sim_tab(self, enable):
        # CustomTkinter não desabilita abas facilmente, 
        # então apenas forçamos a volta se tentar clicar e estiver bloqueado
        if not enable:
            self.tabview.set("Modelagem")
        # (Lógica mais complexa de disable button pode ser adicionada se necessário)

    def run_simulation_logic(self):
        if not self.active_sim: return
        
        # 1. Coleta Constantes Físicas
        user_params = {}
        try:
            for name, entry in self.dynamic_entries.items():
                user_params[name] = float(entry.get())
            # Adiciona g manualmente se não estiver nos inputs
            user_params['g'] = 9.81
            self.active_sim.set_parameters(user_params)
        except ValueError:
            self.log("❌ Erro: Verifique se todos os campos numéricos estão preenchidos corretamente.")
            return

        # 2. Coleta Configuração da Simulação
        try:
            start_pos = [float(x) for x in self.entry_start.get().split(",")]
            end_pos = [float(x) for x in self.entry_end.get().split(",")]
            t_total = float(self.entry_time.get())
            kp = float(self.entry_kp.get())
        except:
            self.log("❌ Erro nos parâmetros de trajetória.")
            return

        self.log("Rodando simulação...")
        
        # 3. Roda (Bloqueante por enquanto, ideal seria thread)
        t, err, tau = self.active_sim.run(t_total, start_pos, end_pos, kp)
        
        # 4. Plota
        self.plot_results(t, err, tau)
        self.log("Simulação finalizada.")

    def plot_results(self, t, err, tau):
        # Limpa plot anterior
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Cria Figura Matplotlib
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        
        # Plot Erro
        ax1.plot(t, err)
        ax1.set_title("Erro nas Juntas (rad)")
        ax1.grid(True)
        ax1.set_ylabel("Erro")
        
        # Plot Torque
        ax2.plot(t, tau)
        ax2.set_title("Torques Aplicados (Nm)")
        ax2.grid(True)
        ax2.set_ylabel("Torque")
        ax2.set_xlabel("Tempo (s)")
        
        plt.tight_layout()

        # Embed no Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==========================================================================
    # JSON (PLACEHOLDERS)
    # ==========================================================================
    def save_json(self):
        self.log("Função Salvar JSON ainda não implementada.")

    def load_json(self):
        self.log("Função Carregar JSON ainda não implementada.")

    def on_closing(self):
        """ Encerra threads e destrói a janela corretamente """
        # Para o loop do Tkinter
        self.quit()
        # Destrói a janela visual
        self.destroy()
        # Força o encerramento do processo Python (mata threads zumbis)
        import sys
        sys.exit()

if __name__ == "__main__":
    app = App()
    app.mainloop()