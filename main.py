import customtkinter as ctk
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.octave import octave_code
import os
import threading
import sys
import numpy as np # <--- ADICIONADO: Faltava isso aqui!
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- IMPORTAÇÕES DOS SEUS MÓDULOS ---
from engine import RobotMathEngine, RobotMathHydro
from simulator import RobotSimulator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hephaestus v4.0 - Integrated Environment")
        self.geometry("1200x850")
        
        # Encerramento seguro
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variáveis de Estado
        self.active_bot = None       
        self.active_sim = None       
        self.joint_rows = []         
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- ABAS ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.tab_model = self.tabview.add("Modelagem")
        self.tab_sim = self.tabview.add("Simulação")
        
        self.setup_modeling_tab()
        self.setup_simulation_tab()
        self.toggle_sim_tab(False)

    def on_closing(self):
        """ Encerra threads e destrói a janela corretamente """
        self.quit()
        self.destroy()
        sys.exit()

    # ==========================================================================
    # ABA 1: MODELAGEM
    # ==========================================================================
    def setup_modeling_tab(self):
        self.tab_model.grid_columnconfigure(0, weight=3)
        self.tab_model.grid_columnconfigure(1, weight=2)
        self.tab_model.grid_rowconfigure(0, weight=1)

        # Esquerda
        left_frame = ctk.CTkFrame(self.tab_model)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        mode_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        mode_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(mode_frame, text="Ambiente:", font=("Arial", 12, "bold")).pack(side="left", padx=5)
        self.mode_var = ctk.StringVar(value="Ar (Seco)")
        self.mode_switch = ctk.CTkSegmentedButton(mode_frame, values=["Ar (Seco)", "Água (UVMS)"], 
                                                  variable=self.mode_var, command=self.update_mode_color)
        self.mode_switch.pack(side="left", padx=10)
        self.update_mode_color("Ar (Seco)")

        self.scroll_joints = ctk.CTkScrollableFrame(left_frame, label_text="Cadeia Cinemática")
        self.scroll_joints.pack(expand=True, fill="both", padx=10, pady=5)
        
        self.add_joint()

        ctrl_joints_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        ctrl_joints_frame.pack(fill="x", padx=10, pady=5)
        btn_add = ctk.CTkButton(ctrl_joints_frame, text="+ Adicionar Junta", command=self.add_joint)
        btn_add.pack(side="left", expand=True, padx=2)
        btn_rem = ctk.CTkButton(ctrl_joints_frame, text="- Remover Última", command=self.remove_joint, fg_color="firebrick")
        btn_rem.pack(side="left", expand=True, padx=2)

        action_frame = ctk.CTkFrame(left_frame)
        action_frame.pack(fill="x", padx=10, pady=10)
        self.btn_calc = ctk.CTkButton(action_frame, text="GERAR MODELO 🚀", command=self.run_modeling, 
                                      height=40, font=ctk.CTkFont(weight="bold"), fg_color="green")
        self.btn_calc.pack(fill="x", padx=10, pady=(10, 5))
        
        # Direita (Log)
        right_frame = ctk.CTkFrame(self.tab_model)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(right_frame, text="Log de Processamento").pack(pady=5)
        self.status_bar = ctk.CTkTextbox(right_frame, font=("Consolas", 12))
        self.status_bar.pack(expand=True, fill="both", padx=5, pady=5)
        self.log("Sistema inicializado.")

    def add_joint(self):
        idx = len(self.joint_rows) + 1
        if idx > 12: return 
        row = ctk.CTkFrame(self.scroll_joints)
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=f"Junta {idx}:", width=50).pack(side="left", padx=5)
        dd = ctk.CTkOptionMenu(row, values=["Rz", "Ry", "Rx", "Dz", "Dy", "Dx"], width=70)
        dd.pack(side="left", padx=5)
        ctk.CTkLabel(row, text="Elo(L):").pack(side="left", padx=5)
        cx = ctk.CTkCheckBox(row, text="X", width=40)
        cx.pack(side="left", padx=2)
        cy = ctk.CTkCheckBox(row, text="Y", width=40)
        cy.pack(side="left", padx=2)
        cz = ctk.CTkCheckBox(row, text="Z", width=40)
        cz.pack(side="left", padx=2)
        if idx == 1: cz.select()
        self.joint_rows.append({"frame": row, "dd": dd, "cx": cx, "cy": cy, "cz": cz})

    def remove_joint(self):
        if len(self.joint_rows) > 1:
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
        print(msg) 

    def run_modeling(self):
        self.btn_calc.configure(state="disabled", text="Calculando...")
        threading.Thread(target=self._run_modeling_thread, daemon=True).start()

    def _run_modeling_thread(self):
        try:
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
            
            if modo == "Água (UVMS)":
                self.active_bot = RobotMathHydro(j_types, l_vecs, logger_callback=self.log)
            else:
                self.active_bot = RobotMathEngine(j_types, l_vecs, logger_callback=self.log)

            results = self.active_bot.run_full_process()
            
            self.log("Compilando equações para o Simulador Numérico...")
            sim_mode = "Hydro" if modo == "Água (UVMS)" else "Air"
            self.active_sim = RobotSimulator(self.active_bot, mode=sim_mode)
            
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
        self.tab_sim.grid_columnconfigure(0, weight=1)
        self.tab_sim.grid_columnconfigure(1, weight=3)
        self.tab_sim.grid_rowconfigure(0, weight=1)

        self.sim_left = ctk.CTkScrollableFrame(self.tab_sim, label_text="Parâmetros")
        self.sim_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(self.sim_left, text="Posição Inicial (x, y, z):").pack(anchor="w")
        self.entry_start = ctk.CTkEntry(self.sim_left)
        self.entry_start.insert(0, "0.5, 0.0, 0.0")
        self.entry_start.pack(fill="x", pady=(0, 5))
        
        self.init_at_start_var = ctk.BooleanVar(value=True)
        self.init_at_start_check = ctk.CTkCheckBox(
            self.sim_left,
            text="Iniciar já na posição inicial",
            variable=self.init_at_start_var
        )
        self.init_at_start_check.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(self.sim_left, text="q_init (rad, opcional):").pack(anchor="w")
        self.entry_q_init = ctk.CTkEntry(self.sim_left)
        self.entry_q_init.insert(0, "")
        self.entry_q_init.pack(fill="x", pady=(0, 5))

        self.use_last_q_var = ctk.BooleanVar(value=True)
        self.use_last_q_check = ctk.CTkCheckBox(
            self.sim_left,
            text="Usar último q convergente",
            variable=self.use_last_q_var
        )
        self.use_last_q_check.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(self.sim_left, text="Posição Final (x, y, z):").pack(anchor="w")
        self.entry_end = ctk.CTkEntry(self.sim_left)
        self.entry_end.insert(0, "0.5, 0.5, 0.2")
        self.entry_end.pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(self.sim_left, text="Tempo Total (s):").pack(anchor="w")
        self.entry_time = ctk.CTkEntry(self.sim_left)
        self.entry_time.insert(0, "5.0")
        self.entry_time.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.sim_left, text="Passo de Física dt (s):").pack(anchor="w")
        self.entry_dt_physics = ctk.CTkEntry(self.sim_left)
        self.entry_dt_physics.insert(0, "0.001")
        self.entry_dt_physics.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.sim_left, text="Passo Visual dt (s):").pack(anchor="w")
        self.entry_dt_visual = ctk.CTkEntry(self.sim_left)
        self.entry_dt_visual.insert(0, "0.05")
        self.entry_dt_visual.pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(self.sim_left, text="--- Controle ---", font=("Arial", 12, "bold")).pack(pady=5)
        self.ctrl_mode_var = ctk.StringVar(value="Torque Computado")
        self.ctrl_mode_dd = ctk.CTkOptionMenu(
            self.sim_left,
            values=["Torque Computado", "ADRC (Robust)", "Sliding Mode (SMC)"],
            variable=self.ctrl_mode_var,
            command=self.update_ctrl_inputs
        )
        self.ctrl_mode_dd.pack(fill="x", pady=(0, 5))

        self.ctrl_inputs_container = ctk.CTkFrame(self.sim_left, fg_color="transparent")
        self.ctrl_inputs_container.pack(fill="x", pady=(0, 10))

        self.ctc_frame = ctk.CTkFrame(self.ctrl_inputs_container)
        ctk.CTkLabel(self.ctc_frame, text="Ganho Kp (ωn²):").pack(anchor="w")
        self.entry_kp = ctk.CTkEntry(self.ctc_frame)
        self.entry_kp.insert(0, "50.0")
        self.entry_kp.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.ctc_frame, text="Fator de amortecimento ζ:").pack(anchor="w")
        self.entry_zeta = ctk.CTkEntry(self.ctc_frame)
        self.entry_zeta.insert(0, "1.0")
        self.entry_zeta.pack(fill="x", pady=(0, 5))

        self.adrc_frame = ctk.CTkFrame(self.ctrl_inputs_container)
        ctk.CTkLabel(self.adrc_frame, text="ωc (rad/s):").pack(anchor="w")
        self.entry_omega_c = ctk.CTkEntry(self.adrc_frame)
        self.entry_omega_c.insert(0, "8.0")
        self.entry_omega_c.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.adrc_frame, text="ωo (rad/s):").pack(anchor="w")
        self.entry_omega_o = ctk.CTkEntry(self.adrc_frame)
        self.entry_omega_o.insert(0, "20.0")
        self.entry_omega_o.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.adrc_frame, text="b0:").pack(anchor="w")
        self.entry_b0 = ctk.CTkEntry(self.adrc_frame)
        self.entry_b0.insert(0, "1.0")
        self.entry_b0.pack(fill="x", pady=(0, 5))

        self.smc_frame = ctk.CTkFrame(self.ctrl_inputs_container)
        ctk.CTkLabel(self.smc_frame, text="Lambda (λ):").pack(anchor="w")
        self.entry_lambda = ctk.CTkEntry(self.smc_frame)
        self.entry_lambda.insert(0, "5.0")
        self.entry_lambda.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.smc_frame, text="Ganho K:").pack(anchor="w")
        self.entry_smc_k = ctk.CTkEntry(self.smc_frame)
        self.entry_smc_k.insert(0, "5.0")
        self.entry_smc_k.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.smc_frame, text="Camada limite ϕ:").pack(anchor="w")
        self.entry_phi = ctk.CTkEntry(self.smc_frame)
        self.entry_phi.insert(0, "0.1")
        self.entry_phi.pack(fill="x", pady=(0, 5))

        self.update_ctrl_inputs(self.ctrl_mode_var.get())

        ctk.CTkLabel(self.sim_left, text="Limite suave dq (rad/s):").pack(anchor="w")
        self.entry_dq_limit = ctk.CTkEntry(self.sim_left)
        self.entry_dq_limit.insert(0, "3.0")
        self.entry_dq_limit.pack(fill="x", pady=(0, 5))

        self.use_feedforward_vel_var = ctk.BooleanVar(value=True)
        self.use_feedforward_vel_check = ctk.CTkCheckBox(
            self.sim_left,
            text="Usar feedforward de velocidade",
            variable=self.use_feedforward_vel_var
        )
        self.use_feedforward_vel_check.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(self.sim_left, text="Perturbação (Nm):").pack(anchor="w")
        self.disturbance_value_label = ctk.CTkLabel(self.sim_left, text="0.0 Nm")
        self.disturbance_value_label.pack(anchor="w")
        self.disturbance_slider = ctk.CTkSlider(
            self.sim_left,
            from_=-20,
            to=20,
            command=self.update_disturbance_label
        )
        self.disturbance_slider.set(0.0)
        self.disturbance_slider.pack(fill="x", pady=(0, 20))

        # === NOVA SEÇÃO: PLANEJAMENTO ===
        ctk.CTkLabel(self.sim_left, text="--- Trajetória ---", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Dropdown Reta/Círculo
        self.traj_type_var = ctk.StringVar(value="Reta")
        self.traj_dd = ctk.CTkOptionMenu(self.sim_left, values=["Reta", "Círculo"], 
                                         variable=self.traj_type_var, command=self.update_traj_inputs)
        self.traj_dd.pack(fill="x", pady=5)

        # Frame para Inputs Específicos do Círculo (Oculto inicialmente)
        self.circle_frame = ctk.CTkFrame(self.sim_left)
        
        ctk.CTkLabel(self.circle_frame, text="Raio (m):").pack(anchor="w")
        self.entry_radius = ctk.CTkEntry(self.circle_frame)
        self.entry_radius.insert(0, "0.3")
        self.entry_radius.pack(fill="x")
        
        ctk.CTkLabel(self.circle_frame, text="Normal (x,y,z):").pack(anchor="w")
        self.entry_normal = ctk.CTkEntry(self.circle_frame)
        self.entry_normal.insert(0, "1, 0, 0") # Plano YZ
        self.entry_normal.pack(fill="x")

        ctk.CTkLabel(self.circle_frame, text="Sentido (+/-):").pack(anchor="w")
        self.switch_dir_var = ctk.StringVar(value="Anti-Horário (+1)")
        self.switch_dir = ctk.CTkSwitch(self.circle_frame, text="Anti-Horário", variable=self.switch_dir_var, 
                                        onvalue="Anti-Horário (+1)", offvalue="Horário (-1)")
        self.switch_dir.pack(pady=5)

        ctk.CTkLabel(self.sim_left, text="--- Constantes Físicas ---", font=("Arial", 12, "bold")).pack(pady=5)
        self.params_container = ctk.CTkFrame(self.sim_left, fg_color="transparent")
        self.params_container.pack(fill="both", expand=True)
        self.dynamic_entries = {}
        self.dynamic_defaults = {}

        self.btn_restore_defaults = ctk.CTkButton(
            self.sim_left,
            text="Restaurar padrões",
            fg_color="#6c757d",
            command=self.restore_sim_defaults
        )
        self.btn_restore_defaults.pack(pady=(10, 5), fill="x")

        self.btn_run_sim = ctk.CTkButton(self.sim_left, text="RODAR SIMULAÇÃO ▶", fg_color="red", command=self.run_simulation_logic)
        self.btn_run_sim.pack(pady=20, side="bottom", fill="x")

        # Container da direita para Gráficos e Animação
        self.sim_right = ctk.CTkFrame(self.tab_sim)
        self.sim_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.plot_frame = ctk.CTkFrame(self.sim_right, fg_color="white")
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Botão para ver Animação 3D (aparece após simular)
        self.btn_anim3d = ctk.CTkButton(self.sim_right, text="VER ANIMAÇÃO 3D 🎥", command=self.play_animation, state="disabled")
        self.btn_anim3d.pack(pady=10)

    # Método auxiliar para mostrar/ocultar inputs do Círculo
    def update_traj_inputs(self, choice):
        if choice == "Círculo":
            self.circle_frame.pack(fill="x", pady=5, after=self.traj_dd)
        else:
            self.circle_frame.pack_forget()

    def update_ctrl_inputs(self, choice):
        for frame in (self.ctc_frame, self.adrc_frame, self.smc_frame):
            frame.pack_forget()
        if choice == "Torque Computado":
            self.ctc_frame.pack(fill="x")
        elif choice == "ADRC (Robust)":
            self.adrc_frame.pack(fill="x")
        elif choice == "Sliding Mode (SMC)":
            self.smc_frame.pack(fill="x")

    def update_disturbance_label(self, value):
        self.disturbance_value_label.configure(text=f"{float(value):.1f} Nm")

    def generate_sim_inputs(self):
        for widget in self.params_container.winfo_children():
            widget.destroy()
        self.dynamic_entries = {}
        self.dynamic_defaults = {}

        if not self.active_sim: return
        ignore_list = set(self.active_bot.q + self.active_bot.dq)

        prefix_configs = [
            {"prefix": "m", "title": "Massas", "unit": "kg", "default": "2.0", "placeholder": "Ex.: 2.0 (kg)"},
            {"prefix": "L", "title": "Comprimentos", "unit": "m", "default": "0.5", "placeholder": "Ex.: 0.5 (m)"},
            {"prefix": "I", "title": "Inércias", "unit": "kg·m²", "default": "0.01", "placeholder": "Ex.: 0.01 (kg·m²)"},
            {"prefix": "rho", "title": "Densidades", "unit": "kg/m³", "default": "1000", "placeholder": "Ex.: 1000 (kg/m³)"},
            {"prefix": "vol", "title": "Volumes", "unit": "m³", "default": "0.005", "placeholder": "Ex.: 0.005 (m³)"},
        ]

        def get_config(sym_name):
            for cfg in prefix_configs:
                if sym_name.startswith(cfg["prefix"]):
                    return cfg
            return {"prefix": "", "title": "Outros", "unit": "", "default": "0.1", "placeholder": "Ex.: 0.1"}

        sections = {}
        for sym in self.active_sim.sym_vars:
            if sym in ignore_list: continue
            if str(sym) in ['t', 'g']: continue

            sym_name = str(sym)
            cfg = get_config(sym_name)
            if cfg["title"] not in sections:
                section_frame = ctk.CTkFrame(self.params_container, fg_color="transparent")
                section_frame.pack(fill="x", pady=(0, 8))
                ctk.CTkLabel(
                    section_frame,
                    text=cfg["title"],
                    font=("Arial", 12, "bold")
                ).pack(anchor="w", pady=(0, 4))
                entries_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
                entries_frame.pack(fill="x")
                sections[cfg["title"]] = entries_frame

            entry_container = sections[cfg["title"]]
            row = ctk.CTkFrame(entry_container)
            row.pack(fill="x", pady=2)
            label_text = f"{sym_name} ({cfg['unit']})" if cfg["unit"] else sym_name
            lbl = ctk.CTkLabel(row, text=label_text, width=120, anchor="w")
            lbl.pack(side="left")
            entry = ctk.CTkEntry(row, placeholder_text=cfg["placeholder"])
            entry.pack(side="right", expand=True, fill="x")
            entry.insert(0, cfg["default"])

            self.dynamic_entries[sym_name] = entry
            self.dynamic_defaults[sym_name] = cfg["default"]

    def restore_sim_defaults(self):
        for name, entry in self.dynamic_entries.items():
            default_value = self.dynamic_defaults.get(name, "")
            entry.delete(0, "end")
            if default_value:
                entry.insert(0, default_value)

    def toggle_sim_tab(self, enable):
        if not enable: self.tabview.set("Modelagem")

    def run_simulation_logic(self):
        """ 
        Gerencia a execução da simulação, lendo inputs da interface 
        para configurar física, postura e trajetória dinâmica.
        """
        if not self.active_sim: 
            self.log("⚠️ Gere o modelo primeiro na aba Modelagem!")
            return
        
        try:
            # ---------------------------------------------------------
            # 1. Parâmetros Físicos (Massas, Inércias, etc.)
            # ---------------------------------------------------------
            user_params = {}
            for name, entry in self.dynamic_entries.items():
                try:
                    val = float(entry.get())
                    user_params[name] = val
                except ValueError:
                    self.log(f"⚠️ Valor inválido para '{name}'. Assumindo 0.0.")
                    user_params[name] = 0.0
            
            user_params['g'] = 9.81
            self.active_sim.set_parameters(user_params)
            
            # ---------------------------------------------------------
            # 2. Configura Postura Preferida (Null Space Control)
            # ---------------------------------------------------------
            # Define 'Home' como zero (ou modifique aqui se quiser 'Elbow Up' fixo)
            self.active_sim.q_home = np.zeros(self.active_sim.num_dof)
            
            # ---------------------------------------------------------
            # 3. Inputs Básicos de Simulação
            # ---------------------------------------------------------
            start_pos = [float(x) for x in self.entry_start.get().split(",")]
            end_pos   = [float(x) for x in self.entry_end.get().split(",")]
            t_total   = float(self.entry_time.get())
            dt_physics = float(self.entry_dt_physics.get())
            dt_visual = float(self.entry_dt_visual.get())
            kp        = float(self.entry_kp.get())
            zeta      = float(self.entry_zeta.get())
            dq_limit  = float(self.entry_dq_limit.get())
            use_feedforward_vel = self.use_feedforward_vel_var.get()
            ctrl_mode = self.ctrl_mode_var.get()

            ctrl_params = {"type": ctrl_mode}
            if ctrl_mode == "ADRC (Robust)":
                omega_c = float(self.entry_omega_c.get())
                omega_o = float(self.entry_omega_o.get())
                b0 = float(self.entry_b0.get())
                kp = omega_c ** 2
                zeta = 1.0
                ctrl_params.update(
                    {
                        "omega_c": omega_c,
                        "omega_o": omega_o,
                        "b0": b0,
                        "kp": kp,
                        "kd": 2 * omega_c,
                        "wo": omega_o,
                        "type": "ADRC",
                    }
                )
            elif ctrl_mode == "Sliding Mode (SMC)":
                lambda_gain = float(self.entry_lambda.get())
                smc_k = float(self.entry_smc_k.get())
                phi = float(self.entry_phi.get())
                ctrl_params.update(
                    {
                        "lambda": lambda_gain,
                        "K": smc_k,
                        "phi": phi,
                        "type": "SMC",
                    }
                )

            if dt_physics <= 0 or dt_visual <= 0:
                self.log("❌ dt_physics e dt_visual devem ser maiores que zero.")
                return

            q_init = None
            q_init_text = self.entry_q_init.get().strip()
            if q_init_text:
                q_init = [float(x) for x in q_init_text.split(",")]
                if len(q_init) != self.active_sim.num_dof:
                    self.log(
                        f"❌ q_init precisa ter {self.active_sim.num_dof} valores."
                    )
                    return
            elif not self.use_last_q_var.get():
                q_init = np.zeros(self.active_sim.num_dof)
            
            # ---------------------------------------------------------
            # 4. Seleção Dinâmica de Trajetória (INTERFACE -> LÓGICA)
            # ---------------------------------------------------------
            # Lê o valor selecionado no Dropdown (Reta ou Círculo)
            mode_str = self.traj_type_var.get()
            
            traj_mode = "Line" # Padrão
            traj_params = {}

            if mode_str == "Círculo":
                traj_mode = "Circle"
                # Lê os campos específicos que aparecem quando "Círculo" é selecionado
                try:
                    r_val = float(self.entry_radius.get())
                    n_vec = [float(x) for x in self.entry_normal.get().split(",")]
                    
                    # Verifica o Switch de sentido
                    # Se o texto conter "Anti", é +1, senão é -1
                    dir_val = 1 if "Anti" in self.switch_dir_var.get() else -1
                    
                    traj_params = {
                        'radius': r_val,
                        'normal': n_vec,
                        'direction': dir_val
                    }
                except ValueError:
                    self.log("❌ Erro nos parâmetros do Círculo. Verifique números e vírgulas.")
                    return

        except ValueError as ve:
            self.log(f"❌ Erro de formatação nos vetores: {ve}")
            return
        except Exception as e:
            self.log(f"❌ Erro crítico na preparação: {e}")
            return

        disturbance_torque = float(self.disturbance_slider.get())

        self.log(f"Iniciando Simulação (Modo: {mode_str}, Controle: {ctrl_mode})...")
        
        # ---------------------------------------------------------
        # 5. Execução
        # ---------------------------------------------------------
        try:
            # Passa os parâmetros lidos para o simulador
            t, err, tau, anim_data = self.active_sim.run(
                t_total, start_pos, end_pos, kp,
                traj_mode=traj_mode, traj_params=traj_params,
                dt_physics=dt_physics, dt_visual=dt_visual,
                init_at_start=self.init_at_start_var.get(),
                zeta=zeta,
                dq_limit=dq_limit,
                use_feedforward_vel=use_feedforward_vel,
                q_init=q_init,
                ctrl_params=ctrl_params,
                disturbance_torque=disturbance_torque
            )
            
            self.last_anim_data = anim_data
            self.last_dt_visual = getattr(self.active_sim, "last_dt_visual", dt_visual)
            self.plot_results(t, err, tau)
            self.log("✅ Simulação finalizada.")
            self.btn_anim3d.configure(state="normal")
            
        except Exception as e:
            self.log(f"❌ Falha na integração numérica: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_results(self, t, err, tau):
        for widget in self.plot_frame.winfo_children(): widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        ax1.plot(t, err); ax1.set_title("Erro (rad)"); ax1.grid(True)

        # Opção 1: Limites Fixos (Ex: de -0.1 a 0.1 rad, aprox 5 graus)
        # Isso vai fazer seu erro de 0.005 parecer uma linha reta (o que é bom visualmente)
        ax1.set_ylim(-0.2, 0.2) 
        
        # OU
        
        # Opção 2: Escala Inteligente (Define um zoom mínimo de 0.05)
        # Se o erro for menor que 0.05, ele mantém a escala em 0.05. Se for maior, ele ajusta.
        # max_err = np.max(np.abs(err))
        # visual_limit = max(max_err * 1.2, 0.05) 
        # ax1.set_ylim(-visual_limit, visual_limit)
        # ---------------------------------------------------------

        ax2.plot(t, tau); ax2.set_title("Torque (Nm)"); ax2.grid(True)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def play_animation(self):
        """ Abre janela 3D """
        if not hasattr(self, 'last_anim_data'): return
        
        import matplotlib.animation as animation
        
        data = self.last_anim_data
        if not data or len(data) == 0:
            self.log("❌ Sem dados de animação.")
            return

        steps = len(data)
        fig = plt.figure("Animação 3D", figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calcula limites
        all_points = []
        for frame in data:
            for p in frame: all_points.append(p)
        all_points = np.array(all_points)
        
        if len(all_points) > 0:
            max_val = np.max(np.abs(all_points)) * 1.2 + 0.1
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_zlim(-max_val, max_val)
        
        line, = ax.plot([], [], [], 'o-', lw=3, markersize=6, color='blue')
        trace, = ax.plot([], [], [], '-', lw=1, color='red', alpha=0.5)
        trace_x, trace_y, trace_z = [], [], []

        def update(frame_idx):
            pose = np.array(data[frame_idx])
            xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
            
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            
            trace_x.append(xs[-1])
            trace_y.append(ys[-1])
            trace_z.append(zs[-1])
            trace.set_data(trace_x, trace_y)
            trace.set_3d_properties(trace_z)
            dt_visual = getattr(self, "last_dt_visual", 0.05)
            ax.set_title(f"T = {frame_idx*dt_visual:.2f}s")
            return line, trace

        ani = animation.FuncAnimation(fig, update, frames=range(0, steps, 1), interval=50, blit=False)
        plt.show()

    def save_json(self): pass
    def load_json(self): pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
