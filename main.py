import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import multiprocessing
import sympy as sp
import sys
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.octave import octave_code
import os
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- IMPORTAÇÕES DOS SEUS MÓDULOS ---
from engine import RobotMathEngine, RobotMathHydro
from simulator import RobotSimulator
from tooltip_utils import RichTooltip, TOOLTIP_CONTENT, phys_tooltip_blocks

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hephaestus v4.0 - Integrated Environment")
        self.geometry("1200x850")
        self.after(10, lambda: self.state("zoomed"))
        
        # Encerramento seguro
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variáveis de Estado
        self.active_bot = None       
        self.active_sim = None       
        self.joint_rows = []
        self.last_sim_results = None  # dict com arrays de resultados + ui_params
        self.comparison_sessions = []  # lista de dicts para comparação de controladores
        self._session_rows = []        # widgets da lista de sessões na aba Análise
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- MENU ---
        self._create_menu()

        # --- ABAS ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.tab_model = self.tabview.add("Modelagem")
        self.tab_sim = self.tabview.add("Simulação")
        self.tab_analysis = self.tabview.add("Análise")
        
        self.setup_modeling_tab()
        self.setup_simulation_tab()
        self.setup_analysis_tab()
        self.toggle_sim_tab(False)
        self._update_menu_state()

    def on_closing(self):
        """ Encerra threads e destrói a janela corretamente """
        if self.active_sim is not None:
            self.active_sim.close()
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
        self.btn_add = ctk.CTkButton(ctrl_joints_frame, text="+ Adicionar Junta", command=self.add_joint)
        self.btn_add.pack(side="left", expand=True, padx=2)
        self.btn_rem = ctk.CTkButton(ctrl_joints_frame, text="- Remover Última", command=self.remove_joint, fg_color="firebrick")
        self.btn_rem.pack(side="left", expand=True, padx=2)

        action_frame = ctk.CTkFrame(left_frame)
        action_frame.pack(fill="x", padx=10, pady=10)
        self.btn_calc = ctk.CTkButton(action_frame, text="GERAR MODELO 🚀", command=self.run_modeling, 
                                      height=40, font=ctk.CTkFont(weight="bold"), fg_color="green")
        self.btn_calc.pack(fill="x", padx=10, pady=(10, 5))

        self._attach_modeling_tooltips()

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

    def _on_auto_b0_toggle(self):
        if self.auto_b0_var.get():
            self.entry_b0.configure(state="disabled")
        else:
            self.entry_b0.configure(state="normal")

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

            # Paralelismo: None = usa todas as CPUs. Se HEPHAESTUS_NO_PARALLEL=1
            # (ex.: executável ainda crashando com ProcessPoolExecutor), força serial.
            num_workers = None
            if os.environ.get("HEPHAESTUS_NO_PARALLEL") == "1":
                num_workers = 1
                self.log("⚠️ HEPHAESTUS_NO_PARALLEL=1: Coriolis em série.")

            if modo == "Água (UVMS)":
                self.active_bot = RobotMathHydro(j_types, l_vecs, logger_callback=self.log, num_workers=num_workers)
            else:
                self.active_bot = RobotMathEngine(j_types, l_vecs, logger_callback=self.log, num_workers=num_workers)

            results = self.active_bot.run_full_process()
            
            # Encerra o executor do modelo anterior antes de criar o novo
            if self.active_sim is not None:
                self.active_sim.close()

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
        self.last_sim_results = None
        self.generate_sim_inputs()
        self.toggle_sim_tab(True)
        self.tabview.set("Simulação")
        self._update_menu_state()
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

        # --- Posição e Tempo (parâmetros essenciais) ---
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
        self.init_at_start_check.pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(self.sim_left, text="Posição Final (x, y, z):").pack(anchor="w")
        self.entry_end = ctk.CTkEntry(self.sim_left)
        self.entry_end.insert(0, "0.5, 0.5, 0.2")
        self.entry_end.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.sim_left, text="Tempo Total (s):").pack(anchor="w")
        self.entry_time = ctk.CTkEntry(self.sim_left)
        self.entry_time.insert(0, "5.0")
        self.entry_time.pack(fill="x", pady=(0, 8))

        # --- Seção Avançada (parâmetros técnicos, oculta por padrão) ---
        self._adv_basic_open = False
        self._btn_adv_basic = ctk.CTkButton(
            self.sim_left, text="▶  Avançado",
            fg_color="gray30", hover_color="gray40", height=28,
            command=self._toggle_adv_basic
        )
        self._btn_adv_basic.pack(fill="x", pady=(0, 8))

        self._adv_basic_frame = ctk.CTkFrame(self.sim_left, fg_color="transparent")

        ctk.CTkLabel(self._adv_basic_frame, text="Passo de Física dt (s):").pack(anchor="w")
        self.entry_dt_physics = ctk.CTkEntry(self._adv_basic_frame)
        self.entry_dt_physics.insert(0, "0.001")
        self.entry_dt_physics.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self._adv_basic_frame, text="Passo Visual dt (s):").pack(anchor="w")
        self.entry_dt_visual = ctk.CTkEntry(self._adv_basic_frame)
        self.entry_dt_visual.insert(0, "0.05")
        self.entry_dt_visual.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self._adv_basic_frame, text="q_init (rad, opcional):").pack(anchor="w")
        self.entry_q_init = ctk.CTkEntry(self._adv_basic_frame)
        self.entry_q_init.insert(0, "")
        self.entry_q_init.pack(fill="x", pady=(0, 5))

        self.use_last_q_var = ctk.BooleanVar(value=True)
        self.use_last_q_check = ctk.CTkCheckBox(
            self._adv_basic_frame,
            text="Usar último q convergente",
            variable=self.use_last_q_var
        )
        self.use_last_q_check.pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(self._adv_basic_frame, text="Limite suave dq (rad/s):").pack(anchor="w")
        self.entry_dq_limit = ctk.CTkEntry(self._adv_basic_frame)
        self.entry_dq_limit.insert(0, "3.0")
        self.entry_dq_limit.pack(fill="x", pady=(0, 5))

        self.use_feedforward_vel_var = ctk.BooleanVar(value=True)
        self.use_feedforward_vel_check = ctk.CTkCheckBox(
            self._adv_basic_frame,
            text="Usar feedforward de velocidade",
            variable=self.use_feedforward_vel_var
        )
        self.use_feedforward_vel_check.pack(anchor="w", pady=(0, 8))

        # --- Controle ---
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

        # CTC
        self.ctc_frame = ctk.CTkFrame(self.ctrl_inputs_container)
        ctk.CTkLabel(self.ctc_frame, text="Ganho Kp (ωn²):").pack(anchor="w")
        self.entry_kp = ctk.CTkEntry(self.ctc_frame)
        self.entry_kp.insert(0, "50.0")
        self.entry_kp.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.ctc_frame, text="Fator de amortecimento ζ:").pack(anchor="w")
        self.entry_zeta = ctk.CTkEntry(self.ctc_frame)
        self.entry_zeta.insert(0, "1.0")
        self.entry_zeta.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.ctc_frame, text="Ganho Integral Ki (0 = PD puro):").pack(anchor="w")
        self.entry_ki = ctk.CTkEntry(self.ctc_frame)
        self.entry_ki.insert(0, "0.0")
        self.entry_ki.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.ctc_frame, text="Anti-windup (limite integral):").pack(anchor="w")
        self.entry_windup = ctk.CTkEntry(self.ctc_frame)
        self.entry_windup.insert(0, "10.0")
        self.entry_windup.pack(fill="x", pady=(0, 5))

        # ADRC
        self.adrc_frame = ctk.CTkFrame(self.ctrl_inputs_container)
        ctk.CTkLabel(self.adrc_frame, text="ωc (rad/s):").pack(anchor="w")
        self.entry_omega_c = ctk.CTkEntry(self.adrc_frame)
        self.entry_omega_c.insert(0, "8.0")
        self.entry_omega_c.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self.adrc_frame, text="ωo (rad/s)  [≥ 5·ωc]:").pack(anchor="w")
        self.entry_omega_o = ctk.CTkEntry(self.adrc_frame)
        self.entry_omega_o.insert(0, "40.0")
        self.entry_omega_o.pack(fill="x", pady=(0, 5))

        self.gravity_ff_var = ctk.BooleanVar(value=True)
        self.chk_gravity_ff = ctk.CTkCheckBox(
            self.adrc_frame, text="FF de gravidade  G(q)",
            variable=self.gravity_ff_var,
        )
        self.chk_gravity_ff.pack(anchor="w", pady=(0, 3))

        self.coriolis_ff_var = ctk.BooleanVar(value=True)
        self.chk_coriolis_ff = ctk.CTkCheckBox(
            self.adrc_frame, text="FF de Coriolis  C(q,dq)",
            variable=self.coriolis_ff_var,
        )
        self.chk_coriolis_ff.pack(anchor="w", pady=(0, 3))

        self.auto_b0_var = ctk.BooleanVar(value=True)
        self.chk_auto_b0 = ctk.CTkCheckBox(
            self.adrc_frame, text="Auto b0  (1/M_ii)",
            variable=self.auto_b0_var,
            command=self._on_auto_b0_toggle,
        )
        self.chk_auto_b0.pack(anchor="w", pady=(0, 3))

        ctk.CTkLabel(self.adrc_frame, text="b0  [≈ 1/M_ii]:").pack(anchor="w")
        self.entry_b0 = ctk.CTkEntry(self.adrc_frame)
        self.entry_b0.insert(0, "1.0")
        self.entry_b0.pack(fill="x", pady=(0, 5))
        self.entry_b0.configure(state="disabled")

        # ADRC avançado (colapsável)
        self._adrc_adv_open = False
        self._btn_adrc_adv = ctk.CTkButton(
            self.adrc_frame, text="▶  Avançado",
            fg_color="gray30", hover_color="gray40", height=26,
            command=self._toggle_adrc_adv
        )
        self._btn_adrc_adv.pack(fill="x", pady=(4, 2))

        self._adrc_adv_frame = ctk.CTkFrame(self.adrc_frame, fg_color="transparent")

        ctk.CTkLabel(self._adrc_adv_frame, text="Limite z (ESO):").pack(anchor="w")
        self.entry_z_limit = ctk.CTkEntry(self._adrc_adv_frame)
        self.entry_z_limit.insert(0, "100.0")
        self.entry_z_limit.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self._adrc_adv_frame, text="Limite τ (Nm):").pack(anchor="w")
        self.entry_tau_limit = ctk.CTkEntry(self._adrc_adv_frame)
        self.entry_tau_limit.insert(0, "50.0")
        self.entry_tau_limit.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self._adrc_adv_frame, text="Máx ωo·dt:").pack(anchor="w")
        self.entry_max_wo_dt = ctk.CTkEntry(self._adrc_adv_frame)
        self.entry_max_wo_dt.insert(0, "0.1")
        self.entry_max_wo_dt.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self._adrc_adv_frame, text="Filtro τ (0-1):").pack(anchor="w")
        self.entry_tau_filter_alpha = ctk.CTkEntry(self._adrc_adv_frame)
        self.entry_tau_filter_alpha.insert(0, "0.8")
        self.entry_tau_filter_alpha.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(self._adrc_adv_frame, text="Filtro z3 (0-1):").pack(anchor="w")
        self.entry_z3_filter_alpha = ctk.CTkEntry(self._adrc_adv_frame)
        self.entry_z3_filter_alpha.insert(0, "0.2")
        self.entry_z3_filter_alpha.pack(fill="x", pady=(0, 5))

        # SMC
        self.smc_frame = ctk.CTkFrame(self.ctrl_inputs_container)
        ctk.CTkLabel(self.smc_frame, text="Variante SMC:").pack(anchor="w")
        self.smc_variant_var = ctk.StringVar(value="CT-SMC")
        self.smc_variant_dd = ctk.CTkOptionMenu(
            self.smc_frame,
            values=["CT-SMC", "Super-Twisting (STA)"],
            variable=self.smc_variant_var,
            command=self._update_smc_variant,
        )
        self.smc_variant_dd.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.smc_frame, text="Lambda (λ):").pack(anchor="w")
        self.entry_lambda = ctk.CTkEntry(self.smc_frame)
        self.entry_lambda.insert(0, "5.0")
        self.entry_lambda.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.smc_frame, text="Filtro q̈_d (α, 0-1):").pack(anchor="w")
        self.entry_ddq_filter_alpha = ctk.CTkEntry(self.smc_frame)
        self.entry_ddq_filter_alpha.insert(0, "1.0")
        self.entry_ddq_filter_alpha.pack(fill="x", pady=(0, 5))

        # Parâmetros exclusivos CT-SMC
        self.smc_ctsmc_frame = ctk.CTkFrame(self.smc_frame, fg_color="transparent")
        ctk.CTkLabel(self.smc_ctsmc_frame, text="Ganho K:").pack(anchor="w")
        self.entry_smc_k = ctk.CTkEntry(self.smc_ctsmc_frame)
        self.entry_smc_k.insert(0, "5.0")
        self.entry_smc_k.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.smc_ctsmc_frame, text="Camada limite ϕ:").pack(anchor="w")
        self.entry_phi = ctk.CTkEntry(self.smc_ctsmc_frame)
        self.entry_phi.insert(0, "0.1")
        self.entry_phi.pack(fill="x", pady=(0, 5))
        self.smc_ctsmc_frame.pack(fill="x")

        # Parâmetros exclusivos Super-Twisting
        self.smc_sta_frame = ctk.CTkFrame(self.smc_frame, fg_color="transparent")
        ctk.CTkLabel(self.smc_sta_frame, text="Ganho k₁ (magnitude):").pack(anchor="w")
        self.entry_sta_k1 = ctk.CTkEntry(self.smc_sta_frame)
        self.entry_sta_k1.insert(0, "5.0")
        self.entry_sta_k1.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.smc_sta_frame, text="Ganho k₂ (integral):").pack(anchor="w")
        self.entry_sta_k2 = ctk.CTkEntry(self.smc_sta_frame)
        self.entry_sta_k2.insert(0, "10.0")
        self.entry_sta_k2.pack(fill="x", pady=(0, 5))

        self.update_ctrl_inputs(self.ctrl_mode_var.get())

        # --- Perturbação ---
        ctk.CTkLabel(self.sim_left, text="Perturbação (Nm):").pack(anchor="w")
        self.disturbance_value_label = ctk.CTkLabel(self.sim_left, text="0.0 Nm")
        self.disturbance_value_label.pack(anchor="w")
        self.disturbance_slider = ctk.CTkSlider(
            self.sim_left, from_=-20, to=20,
            command=self.update_disturbance_label
        )
        self.disturbance_slider.set(0.0)
        self.disturbance_slider.pack(fill="x", pady=(0, 10))

        # --- Trajetória ---
        ctk.CTkLabel(self.sim_left, text="--- Trajetória ---", font=("Arial", 12, "bold")).pack(pady=5)

        self.traj_type_var = ctk.StringVar(value="Reta")
        self.traj_dd = ctk.CTkOptionMenu(
            self.sim_left, values=["Reta", "Círculo"],
            variable=self.traj_type_var, command=self.update_traj_inputs
        )
        self.traj_dd.pack(fill="x", pady=5)

        self.circle_frame = ctk.CTkFrame(self.sim_left)

        ctk.CTkLabel(self.circle_frame, text="Raio (m):").pack(anchor="w")
        self.entry_radius = ctk.CTkEntry(self.circle_frame)
        self.entry_radius.insert(0, "0.3")
        self.entry_radius.pack(fill="x")

        ctk.CTkLabel(self.circle_frame, text="Normal (x,y,z):").pack(anchor="w")
        self.entry_normal = ctk.CTkEntry(self.circle_frame)
        self.entry_normal.insert(0, "1, 0, 0")
        self.entry_normal.pack(fill="x")

        ctk.CTkLabel(self.circle_frame, text="Sentido (+/-):").pack(anchor="w")
        self.switch_dir_var = ctk.StringVar(value="Anti-Horário (+1)")
        self.switch_dir = ctk.CTkSwitch(
            self.circle_frame, text="Anti-Horário",
            variable=self.switch_dir_var,
            onvalue="Anti-Horário (+1)", offvalue="Horário (-1)"
        )
        self.switch_dir.pack(pady=5)

        # --- Orientação ---
        ctk.CTkLabel(self.sim_left, text="--- Orientação ---", font=("Arial", 12, "bold")).pack(pady=5)

        self.orient_mode_var = ctk.StringVar(value="Livre")
        self.orient_mode_dd = ctk.CTkOptionMenu(
            self.sim_left,
            values=[
                "Livre",
                "Fixa",
                "Tangente à Trajetória",
                "Apontar para o Alvo",
                "SLERP",
                "Normal à Superfície",
            ],
            variable=self.orient_mode_var,
            command=self._on_orient_mode_change,
        )
        self.orient_mode_dd.pack(fill="x", pady=(0, 5))

        # SLERP — final orientation as Euler angles (deg, intrinsic XYZ)
        self.slerp_frame = ctk.CTkFrame(self.sim_left)
        ctk.CTkLabel(
            self.slerp_frame,
            text="Orientação Final Rf  (graus, XYZ):",
            anchor="w",
        ).pack(anchor="w", pady=(4, 2))
        for label_text, attr in [("Roll (°):", "entry_slerp_roll"),
                                  ("Pitch (°):", "entry_slerp_pitch"),
                                  ("Yaw (°):", "entry_slerp_yaw")]:
            row = ctk.CTkFrame(self.slerp_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=label_text, width=80, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row)
            e.insert(0, "0.0")
            e.pack(side="left", fill="x", expand=True)
            setattr(self, attr, e)

        # Normal à Superfície — surface normal vector
        self.normal_orient_frame = ctk.CTkFrame(self.sim_left)
        ctk.CTkLabel(
            self.normal_orient_frame,
            text="Normal à Superfície (nx, ny, nz):",
            anchor="w",
        ).pack(anchor="w", pady=(4, 2))
        self.entry_orient_normal = ctk.CTkEntry(self.normal_orient_frame)
        self.entry_orient_normal.insert(0, "0, 0, 1")
        self.entry_orient_normal.pack(fill="x", pady=(0, 4))

        # Shared orientation gain (visible for all non-Livre modes)
        self.orient_gain_frame = ctk.CTkFrame(self.sim_left, fg_color="transparent")
        row_gain = ctk.CTkFrame(self.orient_gain_frame, fg_color="transparent")
        row_gain.pack(fill="x")
        ctk.CTkLabel(row_gain, text="Kp Orient.:", width=90, anchor="w").pack(side="left")
        self.entry_kp_orient = ctk.CTkEntry(row_gain)
        self.entry_kp_orient.insert(0, "5.0")
        self.entry_kp_orient.pack(side="left", fill="x", expand=True)

        # --- Parâmetros Físicos por Elo ---
        ctk.CTkLabel(self.sim_left, text="--- Parâmetros Físicos ---", font=("Arial", 12, "bold")).pack(pady=5)
        self.params_container = ctk.CTkFrame(self.sim_left, fg_color="transparent")
        self.params_container.pack(fill="both", expand=True)
        self.dynamic_entries = {}
        self.dynamic_defaults = {}

        self.btn_restore_defaults = ctk.CTkButton(
            self.sim_left, text="Restaurar padrões",
            fg_color="#6c757d",
            command=self.restore_sim_defaults
        )
        self.btn_restore_defaults.pack(pady=(10, 5), fill="x")

        self.btn_run_sim = ctk.CTkButton(
            self.sim_left, text="RODAR SIMULAÇÃO ▶",
            fg_color="red", command=self.run_simulation_logic
        )
        self.btn_run_sim.pack(pady=20, side="bottom", fill="x")

        # Painel direito
        self.sim_right = ctk.CTkFrame(self.tab_sim)
        self.sim_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.plot_frame = ctk.CTkFrame(self.sim_right, fg_color="white")
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.btn_anim3d = ctk.CTkButton(
            self.sim_right, text="VER ANIMAÇÃO 3D 🎥",
            command=self.play_animation, state="disabled"
        )
        self.btn_anim3d.pack(pady=10)

        self._attach_sim_tooltips()

    # ------------------------------------------------------------------
    def _attach_modeling_tooltips(self):
        """Attach rich tooltips to all modeling-tab widgets."""
        TC = TOOLTIP_CONTENT
        pairs = [
            (self.mode_switch,  TC["mode_switch"]),
            (self.btn_add,      TC["btn_add_joint"]),
            (self.btn_rem,      TC["btn_rem_joint"]),
            (self.btn_calc,     TC["btn_calc"]),
        ]
        for widget, blocks in pairs:
            RichTooltip(widget, blocks)

    # ------------------------------------------------------------------
    def _attach_sim_tooltips(self):
        """Attach rich tooltips to all simulation-tab widgets."""
        TC = TOOLTIP_CONTENT
        pairs = [
            # Posição / tempo
            (self.entry_start,               TC["start_pos"]),
            (self.entry_end,                 TC["end_pos"]),
            (self.init_at_start_check,       TC["init_at_start"]),
            (self.entry_time,                TC["total_time"]),
            # Botão avançado (física)
            (self._btn_adv_basic,            TC["adv_basic_toggle"]),
            # Avançado – física
            (self.entry_dt_physics,          TC["dt_physics"]),
            (self.entry_dt_visual,           TC["dt_visual"]),
            (self.entry_q_init,              TC["q_init"]),
            (self.use_last_q_check,          TC["use_last_q"]),
            (self.entry_dq_limit,            TC["dq_limit"]),
            (self.use_feedforward_vel_check, TC["feedforward_vel"]),
            # Controle – dropdown
            (self.ctrl_mode_dd,              TC["ctrl_mode"]),
            # CTC
            (self.entry_kp,                  TC["kp"]),
            (self.entry_zeta,                TC["zeta"]),
            # ADRC
            (self.entry_omega_c,             TC["omega_c"]),
            (self.entry_omega_o,             TC["omega_o"]),
            (self.chk_gravity_ff,            TC["gravity_ff"]),
            (self.chk_coriolis_ff,           TC["coriolis_ff"]),
            (self.chk_auto_b0,               TC["auto_b0"]),
            (self.entry_b0,                  TC["b0"]),
            (self._btn_adrc_adv,             TC["adrc_adv_toggle"]),
            (self.entry_z_limit,             TC["z_limit"]),
            (self.entry_tau_limit,           TC["tau_limit"]),
            (self.entry_max_wo_dt,           TC["max_wo_dt"]),
            (self.entry_tau_filter_alpha,    TC["tau_filter_alpha"]),
            (self.entry_z3_filter_alpha,     TC["z3_filter_alpha"]),
            # SMC
            (self.entry_lambda,              TC["smc_lambda"]),
            (self.entry_ddq_filter_alpha,    TC["smc_ddq_filter_alpha"]),
            (self.entry_smc_k,               TC["smc_k"]),
            (self.entry_phi,                 TC["smc_phi"]),
            (self.entry_sta_k1,              TC["smc_sta_k1"]),
            (self.entry_sta_k2,              TC["smc_sta_k2"]),
            # Perturbação
            (self.disturbance_slider,        TC["disturbance"]),
            # Trajetória
            (self.traj_dd,                   TC["traj_type"]),
            (self.entry_radius,              TC["radius"]),
            (self.entry_normal,              TC["normal"]),
            (self.switch_dir,                TC["direction"]),
            # Botões de ação
            (self.btn_restore_defaults,      TC["btn_restore_defaults"]),
            (self.btn_run_sim,               TC["btn_run_sim"]),
            (self.btn_anim3d,                TC["btn_anim3d"]),
        ]
        for widget, blocks in pairs:
            RichTooltip(widget, blocks)

    def _toggle_adv_basic(self):
        self._adv_basic_open = not self._adv_basic_open
        if self._adv_basic_open:
            self._adv_basic_frame.pack(fill="x", pady=(0, 8), after=self._btn_adv_basic)
            self._btn_adv_basic.configure(text="▼  Avançado")
        else:
            self._adv_basic_frame.pack_forget()
            self._btn_adv_basic.configure(text="▶  Avançado")

    def _toggle_adrc_adv(self):
        self._adrc_adv_open = not self._adrc_adv_open
        if self._adrc_adv_open:
            self._adrc_adv_frame.pack(fill="x", pady=(0, 5), after=self._btn_adrc_adv)
            self._btn_adrc_adv.configure(text="▼  Avançado")
        else:
            self._adrc_adv_frame.pack_forget()
            self._btn_adrc_adv.configure(text="▶  Avançado")

    # Método auxiliar para mostrar/ocultar inputs do Círculo
    def update_traj_inputs(self, choice):
        if choice == "Círculo":
            self.circle_frame.pack(fill="x", pady=5, after=self.traj_dd)
        else:
            self.circle_frame.pack_forget()

    def _on_orient_mode_change(self, choice):
        """Show/hide orientation auxiliary inputs based on selected mode."""
        self.slerp_frame.pack_forget()
        self.normal_orient_frame.pack_forget()
        self.orient_gain_frame.pack_forget()

        if choice == "Livre":
            return

        # Anchor: pack the gain row directly after the dropdown, then insert
        # mode-specific frames before it so they appear between dd and gain.
        self.orient_gain_frame.pack(fill="x", pady=(0, 8), after=self.orient_mode_dd)

        if choice == "SLERP":
            self.slerp_frame.pack(fill="x", pady=(0, 4), after=self.orient_mode_dd)
        elif choice == "Normal à Superfície":
            self.normal_orient_frame.pack(fill="x", pady=(0, 4), after=self.orient_mode_dd)

    def _update_smc_variant(self, choice):
        if choice == "CT-SMC":
            self.smc_sta_frame.pack_forget()
            self.smc_ctsmc_frame.pack(fill="x")
        else:
            self.smc_ctsmc_frame.pack_forget()
            self.smc_sta_frame.pack(fill="x")

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

        if not self.active_sim:
            return

        sym_names = {str(s) for s in self.active_sim.sym_vars}
        n = len(self.active_bot.joint_config)

        # ---------- helper para criar um entry bloqueado (off-diagonal / eixo inativo) ----------
        def _locked_entry(parent, value="0"):
            e = ctk.CTkEntry(parent, width=62)
            e.insert(0, value)
            e.configure(state="disabled")
            return e

        # ---------- helper para criar um entry editável e registrá-lo ----------
        def _reg_entry(parent, key, default):
            e = ctk.CTkEntry(parent, width=62)
            e.insert(0, default)
            self.dynamic_entries[key] = e
            self.dynamic_defaults[key] = default
            return e

        # ====================================================================
        # Seções por elo
        # ====================================================================
        for i in range(n):
            link_num = i + 1
            mask = self.active_bot.link_vectors_mask[i]   # e.g. [0, 0, 1]

            section = ctk.CTkFrame(self.params_container, border_width=1)
            section.pack(fill="x", pady=(0, 8), padx=2)
            ctk.CTkLabel(
                section, text=f"Elo {link_num}",
                font=("Arial", 12, "bold")
            ).pack(anchor="w", padx=8, pady=(6, 4))

            inner = ctk.CTkFrame(section, fg_color="transparent")
            inner.pack(fill="x", padx=8, pady=(0, 8))

            # --- Massa ---
            mass_key = f"m{link_num}"
            if mass_key in sym_names:
                row = ctk.CTkFrame(inner, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text="Massa (kg):", width=160, anchor="w").pack(side="left")
                _reg_entry(row, mass_key, "2.0").pack(side="left")

            # --- Comprimento como vetor [Lx, Ly, Lz] ---
            L_key = f"L{link_num}"
            if L_key in sym_names:
                row = ctk.CTkFrame(inner, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text="Comprimento (m):", width=160, anchor="w").pack(side="left")
                axis_labels = ["x", "y", "z"]
                active_assigned = False
                for j, ax in enumerate(axis_labels):
                    ctk.CTkLabel(row, text=ax, width=14, anchor="e").pack(side="left")
                    if int(mask[j]) and not active_assigned:
                        _reg_entry(row, L_key, "0.5").pack(side="left", padx=(0, 6))
                        active_assigned = True
                    else:
                        _locked_entry(row).pack(side="left", padx=(0, 6))
                # fallback: se mask for todo zero, registra mesmo assim
                if not active_assigned and L_key not in self.dynamic_entries:
                    _reg_entry(row, L_key, "0.5").pack(side="left", padx=(0, 6))

            # --- Centro de massa como vetor [cx, cy, cz] ---
            cx_key = f"cx{link_num}"
            if cx_key in sym_names:
                row = ctk.CTkFrame(inner, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text="Centro de massa (m):", width=160, anchor="w").pack(side="left")
                for sym_key, ax in [
                    (f"cx{link_num}", "x"),
                    (f"cy{link_num}", "y"),
                    (f"cz{link_num}", "z"),
                ]:
                    ctk.CTkLabel(row, text=ax, width=14, anchor="e").pack(side="left")
                    _reg_entry(row, sym_key, "0.0").pack(side="left", padx=(0, 6))

            # --- Tensor de inércia como matriz 3×3 ---
            Ixx_key = f"Ixx{link_num}"
            if Ixx_key in sym_names:
                ctk.CTkLabel(
                    inner, text="Tensor de inércia (kg·m²):", anchor="w"
                ).pack(anchor="w", pady=(6, 2))

                diag_keys = [f"Ixx{link_num}", f"Iyy{link_num}", f"Izz{link_num}"]
                diag_labels = ["Ixx", "Iyy", "Izz"]

                mat_frame = ctk.CTkFrame(inner, fg_color="transparent")
                mat_frame.pack(anchor="w")

                # Cabeçalho de colunas
                header = ctk.CTkFrame(mat_frame, fg_color="transparent")
                header.pack()
                ctk.CTkLabel(header, text="", width=18).pack(side="left")
                for ax in ["x", "y", "z"]:
                    ctk.CTkLabel(header, text=ax, width=68, anchor="center").pack(side="left", padx=2)

                for r in range(3):
                    mat_row = ctk.CTkFrame(mat_frame, fg_color="transparent")
                    mat_row.pack()
                    ctk.CTkLabel(mat_row, text=diag_labels[r][1], width=18, anchor="e").pack(side="left")
                    for c in range(3):
                        if r == c:
                            _reg_entry(mat_row, diag_keys[r], "0.01").pack(side="left", padx=2, pady=1)
                        else:
                            _locked_entry(mat_row).pack(side="left", padx=2, pady=1)

            # --- Volume (modo Hidro) ---
            vol_key = f"vol{link_num}"
            if vol_key in sym_names:
                row = ctk.CTkFrame(inner, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text="Volume (m³):", width=160, anchor="w").pack(side="left")
                _reg_entry(row, vol_key, "0.005").pack(side="left")

            # --- Massas adicionadas (modo Hidro) ---
            ma_u_key = f"ma_u{link_num}"
            if ma_u_key in sym_names:
                ctk.CTkLabel(
                    inner, text="Massa adicionada (kg / kg·m²):", anchor="w"
                ).pack(anchor="w", pady=(6, 2))
                ma_linear  = [f"ma_u{link_num}", f"ma_v{link_num}", f"ma_w{link_num}"]
                ma_angular = [f"ma_p{link_num}", f"ma_q{link_num}", f"ma_r{link_num}"]
                for group, axes, default in [
                    (ma_linear,  ["u", "v", "w"], "5.0"),
                    (ma_angular, ["p", "q", "r"], "0.1"),
                ]:
                    row = ctk.CTkFrame(inner, fg_color="transparent")
                    row.pack(fill="x", pady=1)
                    for sym_key, ax in zip(group, axes):
                        if sym_key in sym_names:
                            ctk.CTkLabel(row, text=ax, width=14, anchor="e").pack(side="left")
                            _reg_entry(row, sym_key, default).pack(side="left", padx=(0, 6))

            # --- Arrasto hidrodinâmico (modo Hidro) ---
            if vol_key in sym_names:
                ctk.CTkLabel(
                    inner, text="Arrasto hidrodinâmico:", anchor="w"
                ).pack(anchor="w", pady=(6, 2))
                row = ctk.CTkFrame(inner, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text="D_lin (Ns/m):", width=160, anchor="w").pack(side="left")
                _reg_entry(row, f"d_lin{link_num}", "0.0").pack(side="left")
                row2 = ctk.CTkFrame(inner, fg_color="transparent")
                row2.pack(fill="x", pady=2)
                ctk.CTkLabel(row2, text="D_quad (Ns²/m²):", width=160, anchor="w").pack(side="left")
                _reg_entry(row2, f"d_quad{link_num}", "0.0").pack(side="left")

            # --- Atrito nas juntas (sempre visível) ---
            ctk.CTkLabel(
                inner, text="Atrito na junta:", anchor="w"
            ).pack(anchor="w", pady=(6, 2))
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text="Bv (Nms/rad):", width=160, anchor="w").pack(side="left")
            _reg_entry(row, f"bv{link_num}", "0.0").pack(side="left")
            row2 = ctk.CTkFrame(inner, fg_color="transparent")
            row2.pack(fill="x", pady=2)
            ctk.CTkLabel(row2, text="Fc (Nm, Coulomb):", width=160, anchor="w").pack(side="left")
            _reg_entry(row2, f"fc{link_num}", "0.0").pack(side="left")

        # ====================================================================
        # Parâmetros globais (rho para modo Hidro)
        # ====================================================================
        global_params = {"rho": ("Densidade do fluido (kg/m³):", "1000")}
        has_globals = any(k in sym_names for k in global_params)
        if has_globals:
            section = ctk.CTkFrame(self.params_container, border_width=1)
            section.pack(fill="x", pady=(0, 8), padx=2)
            ctk.CTkLabel(
                section, text="Ambiente",
                font=("Arial", 12, "bold")
            ).pack(anchor="w", padx=8, pady=(6, 4))
            inner = ctk.CTkFrame(section, fg_color="transparent")
            inner.pack(fill="x", padx=8, pady=(0, 8))
            for k, (label, default) in global_params.items():
                if k in sym_names:
                    row = ctk.CTkFrame(inner, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    ctk.CTkLabel(row, text=label, width=200, anchor="w").pack(side="left")
                    _reg_entry(row, k, default).pack(side="left")

        # Attach tooltips to all dynamically created physical-parameter entries
        for key, entry in self.dynamic_entries.items():
            blocks = phys_tooltip_blocks(key)
            if blocks:
                RichTooltip(entry, blocks)

    def restore_sim_defaults(self):
        for name, entry in self.dynamic_entries.items():
            default_value = self.dynamic_defaults.get(name, "")
            was_disabled = str(entry.cget("state")) == "disabled"
            if was_disabled:
                entry.configure(state="normal")
            entry.delete(0, "end")
            if default_value:
                entry.insert(0, default_value)
            if was_disabled:
                entry.configure(state="disabled")

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

            # ---------------------------------------------------------
            # 3a. Parâmetros de Atrito e Arrasto por junta
            # ---------------------------------------------------------
            n_dof = self.active_sim.num_dof
            _Bv    = [float(user_params.get(f"bv{i+1}",     0.0)) for i in range(n_dof)]
            _Fc    = [float(user_params.get(f"fc{i+1}",     0.0)) for i in range(n_dof)]
            _D_lin  = [float(user_params.get(f"d_lin{i+1}", 0.0)) for i in range(n_dof)]
            _D_quad = [float(user_params.get(f"d_quad{i+1}",0.0)) for i in range(n_dof)]

            friction_params_run = {"Bv": _Bv, "Fc": _Fc, "epsilon": 0.05}
            drag_params_run     = {"D_lin": _D_lin, "D_quad": _D_quad}

            ctrl_params = {"type": ctrl_mode}
            if ctrl_mode == "Torque Computado":
                try:
                    ki_val     = float(self.entry_ki.get())
                    windup_val = float(self.entry_windup.get())
                except ValueError:
                    ki_val, windup_val = 0.0, 10.0
                ctrl_params.update({"ki": ki_val, "windup_limit": windup_val})
                if ki_val > 0:
                    self.log(f"ℹ️ CTC-PID ativo: Ki={ki_val:.3g}, anti-windup={windup_val:.3g}")
            elif ctrl_mode == "ADRC (Robust)":
                omega_c = float(self.entry_omega_c.get())
                omega_o = float(self.entry_omega_o.get())
                auto_b0 = self.auto_b0_var.get()
                b0 = float(self.entry_b0.get()) if not auto_b0 else 1.0
                z_limit = float(self.entry_z_limit.get())
                tau_limit = float(self.entry_tau_limit.get())
                max_wo_dt = float(self.entry_max_wo_dt.get())
                tau_filter_alpha = float(self.entry_tau_filter_alpha.get())
                z3_filter_alpha = float(self.entry_z3_filter_alpha.get())
                if omega_c <= 0 or omega_o <= 0 or b0 <= 0:
                    self.log("❌ omega_c, omega_o e b0 devem ser maiores que zero.")
                    return
                if z_limit <= 0 or tau_limit <= 0 or max_wo_dt <= 0:
                    self.log("❌ z_limit, tau_limit e max_wo_dt devem ser maiores que zero.")
                    return
                if tau_filter_alpha <= 0 or tau_filter_alpha > 1:
                    self.log("❌ tau_filter_alpha deve estar entre 0 e 1.")
                    return
                if not (0.0 < z3_filter_alpha <= 1.0):
                    self.log("❌ z3_filter_alpha deve estar entre 0 (exclusive) e 1.")
                    return
                if omega_o < 3 * omega_c:
                    self.log(f"⚠️ ωo ({omega_o}) < 3·ωc ({3*omega_c:.1f}). Recomendado ωo ≥ 5·ωc para ESO convergir antes do controlador.")
                elif omega_o < 5 * omega_c:
                    self.log(f"⚠️ ωo/ωc = {omega_o/omega_c:.1f} (recomendado ≥ 5). Rastreamento do ESO pode ser lento.")
                gravity_ff  = self.gravity_ff_var.get()
                coriolis_ff = self.coriolis_ff_var.get()
                if auto_b0:
                    self.log("ℹ️ b0 será estimado automaticamente como 1/M_ii na postura inicial.")
                if gravity_ff and coriolis_ff:
                    self.log("ℹ️ FF G(q)+C(q,dq) ativo: z₃ estimará apenas distúrbios externos.")
                elif gravity_ff:
                    self.log("ℹ️ FF G(q) ativo: z₃ estimará Coriolis + distúrbios externos.")
                kp = omega_c ** 2
                zeta = 1.0
                ctrl_params.update(
                    {
                        "omega_c": omega_c,
                        "omega_o": omega_o,
                        "b0": b0,
                        "auto_b0": auto_b0,
                        "gravity_ff":  gravity_ff,
                        "coriolis_ff": coriolis_ff,
                        "kp": kp,
                        "kd": 2 * omega_c,
                        "wo": omega_o,
                        "z_limit": z_limit,
                        "tau_limit": tau_limit,
                        "max_wo_dt": max_wo_dt,
                        "tau_filter_alpha": tau_filter_alpha,
                        "z3_filter_alpha": z3_filter_alpha,
                        "type": "ADRC",
                    }
                )
            elif ctrl_mode == "Sliding Mode (SMC)":
                lambda_gain = float(self.entry_lambda.get())
                ddq_alpha = float(self.entry_ddq_filter_alpha.get())
                ddq_alpha = max(1e-6, min(1.0, ddq_alpha))
                smc_variant = self.smc_variant_var.get()
                if smc_variant == "Super-Twisting (STA)":
                    sta_k1 = float(self.entry_sta_k1.get())
                    sta_k2 = float(self.entry_sta_k2.get())
                    if lambda_gain <= 0 or sta_k1 <= 0 or sta_k2 <= 0:
                        self.log("❌ lambda, k1 e k2 devem ser maiores que zero.")
                        return
                    ctrl_params.update(
                        {
                            "lambda": lambda_gain,
                            "k1": sta_k1,
                            "k2": sta_k2,
                            "ddq_filter_alpha": ddq_alpha,
                            "type": "STA",
                        }
                    )
                else:
                    smc_k = float(self.entry_smc_k.get())
                    phi = float(self.entry_phi.get())
                    if lambda_gain <= 0 or smc_k <= 0:
                        self.log("❌ lambda e K devem ser maiores que zero.")
                        return
                    ctrl_params.update(
                        {
                            "lambda": lambda_gain,
                            "K": smc_k,
                            "phi": phi,
                            "ddq_filter_alpha": ddq_alpha,
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
                try:
                    r_val = float(self.entry_radius.get())
                    n_vec = [float(x) for x in self.entry_normal.get().split(",")]
                    dir_val = 1 if "Anti" in self.switch_dir_var.get() else -1
                    traj_params = {
                        'radius': r_val,
                        'normal': n_vec,
                        'direction': dir_val
                    }
                except ValueError:
                    self.log("❌ Erro nos parâmetros do Círculo. Verifique números e vírgulas.")
                    return

            # ---------------------------------------------------------
            # 5. Orientação
            # ---------------------------------------------------------
            orient_mode_str = self.orient_mode_var.get()
            orient_params = {}

            if orient_mode_str == "SLERP":
                try:
                    roll_deg  = float(self.entry_slerp_roll.get())
                    pitch_deg = float(self.entry_slerp_pitch.get())
                    yaw_deg   = float(self.entry_slerp_yaw.get())
                    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
                    cr, sr = np.cos(roll),  np.sin(roll)
                    cp, sp = np.cos(pitch), np.sin(pitch)
                    cy, sy = np.cos(yaw),   np.sin(yaw)
                    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                    orient_params['Rf'] = Rz @ Ry @ Rx
                except ValueError:
                    self.log("❌ Ângulos SLERP inválidos. Use valores numéricos.")
                    return

            elif orient_mode_str == "Normal à Superfície":
                try:
                    n_vals = [float(x) for x in self.entry_orient_normal.get().split(",")]
                    if len(n_vals) != 3:
                        raise ValueError("Normal deve ter exatamente 3 componentes.")
                    orient_params['normal'] = n_vals
                except ValueError as e:
                    self.log(f"❌ Vetor normal inválido: {e}")
                    return

            if orient_mode_str != "Livre":
                try:
                    orient_params['Kp_orient'] = float(self.entry_kp_orient.get())
                except ValueError:
                    orient_params['Kp_orient'] = 5.0

        except ValueError as ve:
            self.log(f"❌ Erro de formatação nos vetores: {ve}")
            return
        except Exception as e:
            self.log(f"❌ Erro crítico na preparação: {e}")
            return

        dist_value = float(self.disturbance_slider.get())

        ctrl_label = ctrl_params.get("type", ctrl_mode)
        self.log(
            "Iniciando Simulação "
            f"(Trajetória: {mode_str}, Orientação: {orient_mode_str}, "
            f"Controle: {ctrl_label}, Perturbação: {dist_value:.1f} Nm)..."
        )
        
        # ---------------------------------------------------------
        # 5. Execução
        # ---------------------------------------------------------
        try:
            # Passa os parâmetros lidos para o simulador
            t, err, tau, anim_data, elapsed_time = self.active_sim.run(
                t_total, start_pos, end_pos, kp,
                traj_mode=traj_mode, traj_params=traj_params,
                dt_physics=dt_physics, dt_visual=dt_visual,
                init_at_start=self.init_at_start_var.get(),
                zeta=zeta,
                dq_limit=dq_limit,
                use_feedforward_vel=use_feedforward_vel,
                q_init=q_init,
                ctrl_params=ctrl_params,
                disturbance_torque=dist_value,
                orient_mode=orient_mode_str,
                orient_params=orient_params,
                friction_params=friction_params_run,
                drag_params=drag_params_run,
            )
            
            self.last_anim_data = anim_data
            self.last_dt_visual = getattr(self.active_sim, "last_dt_visual", dt_visual)
            self.last_sim_results = {
                "t":            t,
                "err":          err,
                "tau":          tau,
                "anim_data":    anim_data,
                "dt_visual":    self.last_dt_visual,
                "ui_params":    self._collect_ui_params(),
                "elapsed_time": elapsed_time,
            }
            self.plot_results(t, err, tau)
            self._update_menu_state()
            self.log(f"✅ Simulação finalizada em {elapsed_time:.2f}s.")
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

    def _is_free_floating(self):
        """Retorna True se o robô não tem comprimento de elo (drone/corpo livre)."""
        if self.active_bot is None:
            return False
        try:
            total = sum(
                int(self.active_bot.link_vectors_mask[i][j])
                for i in range(len(self.active_bot.link_vectors_mask))
                for j in range(3)
            )
            return total == 0
        except Exception:
            return False

    @staticmethod
    def _build_auv_geometry(size):
        """Pré-computa toda a geometria do AUV no frame local (X=frente, Z=cima).
        Retorna dict de arrays (N×3) prontos para transformação."""
        L   = size * 1.4   # semi-comprimento do casco
        r   = size * 0.28  # raio máximo do casco
        NaN = np.full((1, 3), np.nan)

        # --- Perfil do casco (curva torpedo) ---
        n = 60
        s = np.linspace(0, 1, n)
        x = (s - 0.5) * 2 * L

        def r_at(sv):
            if sv < 0.12:  return r * (sv / 0.12) ** 0.55
            elif sv > 0.80: return r * ((1 - sv) / 0.20) ** 0.40
            else:           return r

        rv = np.array([r_at(si) for si in s])

        top_prof  = np.column_stack([x,  np.zeros(n),  rv])
        bot_prof  = np.column_stack([x,  np.zeros(n), -rv])
        port_prof = np.column_stack([x,  rv, np.zeros(n)])
        stbd_prof = np.column_stack([x, -rv, np.zeros(n)])

        # --- Anéis transversais (3: ré, meio, vante) ---
        th = np.linspace(0, 2 * np.pi, 25)
        def ring(xv, rv2):
            return np.column_stack([np.full(25, xv), rv2 * np.cos(th), rv2 * np.sin(th)])

        rings = np.vstack([
            ring(-0.55 * L, r_at(0.23)),  NaN,
            ring( 0.0,       r),           NaN,
            ring( 0.55 * L, r_at(0.78)),
        ])

        # --- Torre de comando (casco de torpedo não tem; usamos janela/dome) ---
        # Viewport: semi-anel na proa
        th_vp  = np.linspace(-np.pi / 2, np.pi / 2, 18)
        rv_vp  = r_at(0.90)
        xv_bow = 0.80 * L
        vp_h   = np.column_stack([np.full(18, xv_bow),
                                   rv_vp * np.cos(th_vp),
                                   rv_vp * np.sin(th_vp)])
        vp_v   = np.column_stack([np.full(18, xv_bow),
                                   rv_vp * np.sin(th_vp),   # rotacionado 90°
                                   rv_vp * np.cos(th_vp)])
        viewport = np.vstack([vp_h, NaN, vp_v])

        # --- Superfícies de controle na popa (X em forma de cruz) ---
        fin_xr  = -0.70 * L   # raiz da aleta
        fin_xt  = -0.92 * L   # ponta da aleta
        fin_zr  =  r * 0.88   # raiz (sobre o casco)
        fin_zt  =  r * 1.55   # ponta
        # Sweep: ponta mais para trás
        fins_list = []
        for sign_y, sign_z in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            yr = sign_y * fin_zr;  yt = sign_y * fin_zt
            zr = sign_z * fin_zr;  zt = sign_z * fin_zt
            # Triângulo da aleta (raiz–ponta–base sweep)
            fins_list.append(np.array([
                [fin_xr, yr, zr],
                [fin_xt, yt, zt],
                [fin_xt, yt * 0.5, zt * 0.5],
                [fin_xr, yr, zr],
            ]))
        fins = np.vstack([f if i == 0 else np.vstack([NaN, f])
                          for i, f in enumerate(fins_list)])

        # --- Hélice na popa ---
        th_p   = np.linspace(0, 2 * np.pi, 28)
        r_prop = r * 0.65
        x_prop = -L * 1.02
        prop_ring = np.column_stack([np.full(28, x_prop),
                                     r_prop * np.cos(th_p),
                                     r_prop * np.sin(th_p)])
        # 3 pás
        blade_angles = [0, np.pi * 2 / 3, np.pi * 4 / 3]
        blades_list  = [
            np.array([[x_prop, 0, 0],
                       [x_prop, r_prop * np.cos(a), r_prop * np.sin(a)]])
            for a in blade_angles
        ]
        blades = np.vstack([b if i == 0 else np.vstack([NaN, b])
                            for i, b in enumerate(blades_list)])
        propeller = np.vstack([prop_ring, NaN, blades])

        return {
            "top":      top_prof,
            "bot":      bot_prof,
            "port":     port_prof,
            "stbd":     stbd_prof,
            "rings":    rings,
            "viewport": viewport,
            "fins":     fins,
            "prop":     propeller,
        }

    def play_animation(self):
        """Abre janela 3D — modo AUV/Submarino, Drone ou Braço conforme configuração."""
        if not hasattr(self, 'last_anim_data'):
            return

        import matplotlib.animation as animation

        data = self.last_anim_data
        if not data or len(data) == 0:
            self.log("❌ Sem dados de animação.")
            return

        is_new_format  = isinstance(data[0], dict)
        is_free        = self._is_free_floating()
        is_underwater  = (self.mode_var.get() == "Água (UVMS)")
        steps          = len(data)
        dt_visual      = getattr(self, "last_dt_visual", 0.05)

        # Detecta UVMS: veículo (DOFs sem elo) + braço (DOFs com elo)
        vehicle_dof = getattr(self.active_sim, 'vehicle_dof', 0) if self.active_sim else 0
        total_dof   = len(self.active_bot.joint_config) if self.active_bot else 0
        is_uvms     = (is_new_format and is_underwater
                       and 0 < vehicle_dof < total_dof
                       and "R_vehicle" in data[0])

        fig = plt.figure("Animação 3D", figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

        # Helper interno reutilizado por AUV e UVMS
        def _apply_tf(pts_local, R, pos):
            mask  = np.any(np.isnan(pts_local), axis=1)
            out   = pts_local.copy()
            valid = ~mask
            out[valid] = (R @ pts_local[valid].T).T + pos
            return out

        def _set_line(line_obj, pts):
            line_obj.set_data(pts[:, 0], pts[:, 1])
            line_obj.set_3d_properties(pts[:, 2])

        def _setup_underwater(ax_ref, fig_ref):
            ax_ref.set_facecolor('#0a1628')
            fig_ref.patch.set_facecolor('#0a1628')
            ax_ref.tick_params(colors='#8ab4d8')
            for lbl in (ax_ref.xaxis.label, ax_ref.yaxis.label, ax_ref.zaxis.label):
                lbl.set_color('#8ab4d8')

        # ------------------------------------------------------------------ #
        #  MODO UVMS  (AUV + braço acoplado, ambiente água)                  #
        # ------------------------------------------------------------------ #
        if is_uvms:
            # Posições do veículo e do end-effector
            veh_pts = np.array([np.array(f["links"][vehicle_dof], dtype=float) for f in data])
            ee_pts  = np.array([np.array(f["links"][-1],          dtype=float) for f in data])
            all_pts = np.vstack([veh_pts, ee_pts])
            c       = all_pts.mean(axis=0)
            spread  = max(np.max(np.abs(all_pts - c)) * 1.8 + 0.05, 0.3)
            ax.set_xlim(c[0]-spread, c[0]+spread)
            ax.set_ylim(c[1]-spread, c[1]+spread)
            ax.set_zlim(c[2]-spread, c[2]+spread)
            _setup_underwater(ax, fig)

            size      = spread * 0.16
            TRAIL_LEN = 120
            geom      = self._build_auv_geometry(size)

            C_HULL  = '#4a90c4';  C_RING  = '#2d6a9f';  C_FIN   = '#5ba3d9'
            C_PROP  = '#f0a500';  C_TRAIL = '#00e5ff';  C_ARM   = '#f39c12'
            C_JOINT = '#e74c3c';  C_AX_X  = '#ff4d4d';  C_AX_Y  = '#66ff66'
            C_AX_Z  = '#4db8ff'

            def _mk(color, lw, alpha=1.0):
                return ax.plot([], [], [], '-', lw=lw, color=color, alpha=alpha)[0]

            # AUV body lines
            hull_top   = _mk(C_HULL,   1.8)
            hull_bot   = _mk(C_HULL,   1.8)
            hull_port  = _mk(C_HULL,   1.1, 0.65)
            hull_stbd  = _mk(C_HULL,   1.1, 0.65)
            rings_line = _mk(C_RING,   0.9, 0.60)
            vp_line    = _mk('#a8d8f0', 1.3, 0.85)
            fins_line  = _mk(C_FIN,    1.5)
            prop_line  = _mk(C_PROP,   1.4, 0.85)
            # Arm lines
            arm_line,  = ax.plot([], [], [], 'o-', lw=2.5, color=C_ARM,   ms=7,
                                 markerfacecolor=C_JOINT, markeredgecolor='white',
                                 markeredgewidth=0.8)
            trail_line = _mk(C_TRAIL, 1.4, 0.50)
            # Orientation axes at end-effector
            axis_x = _mk(C_AX_X, 1.8)
            axis_y = _mk(C_AX_Y, 1.8)
            axis_z = _mk(C_AX_Z, 1.8)

            ax.scatter(*veh_pts[0],  c='#00e676', s=50, zorder=5,
                       label='Início (veículo)', depthshade=False)
            ax.scatter(*ee_pts[-1],  c='#ff1744', s=55, zorder=5, marker='*',
                       label='Alvo (efetuador)', depthshade=False)
            ax.legend(loc='upper left', fontsize=8, facecolor='#0a1628',
                      labelcolor='#8ab4d8', edgecolor='#2d6a9f')

            trail_x, trail_y, trail_z = [], [], []

            def _update_uvms(fi):
                frame   = data[fi]
                links   = [np.array(p, dtype=float) for p in frame["links"]]
                veh_pos = links[vehicle_dof]
                ee_pos  = links[-1]
                R_veh   = np.array(frame["R_vehicle"], dtype=float).reshape(3, 3)
                R_end   = np.array(frame["R"],         dtype=float).reshape(3, 3)

                # AUV body (submarino no frame do veículo)
                _set_line(hull_top,   _apply_tf(geom["top"],      R_veh, veh_pos))
                _set_line(hull_bot,   _apply_tf(geom["bot"],      R_veh, veh_pos))
                _set_line(hull_port,  _apply_tf(geom["port"],     R_veh, veh_pos))
                _set_line(hull_stbd,  _apply_tf(geom["stbd"],     R_veh, veh_pos))
                _set_line(rings_line, _apply_tf(geom["rings"],    R_veh, veh_pos))
                _set_line(vp_line,    _apply_tf(geom["viewport"], R_veh, veh_pos))
                _set_line(fins_line,  _apply_tf(geom["fins"],     R_veh, veh_pos))
                _set_line(prop_line,  _apply_tf(geom["prop"],     R_veh, veh_pos))

                # Braço: cadeia do veículo ao end-effector
                arm_pts = np.array(links[vehicle_dof:])
                arm_line.set_data(arm_pts[:, 0], arm_pts[:, 1])
                arm_line.set_3d_properties(arm_pts[:, 2])

                # Trilha do end-effector
                trail_x.append(ee_pos[0])
                trail_y.append(ee_pos[1])
                trail_z.append(ee_pos[2])
                trail_line.set_data(trail_x[-TRAIL_LEN:], trail_y[-TRAIL_LEN:])
                trail_line.set_3d_properties(trail_z[-TRAIL_LEN:])

                # Eixos de orientação do efetuador final
                ax_len = size * 0.9
                for ln, col in [(axis_x, 0), (axis_y, 1), (axis_z, 2)]:
                    tip = ee_pos + R_end[:, col] * ax_len
                    ln.set_data([ee_pos[0], tip[0]], [ee_pos[1], tip[1]])
                    ln.set_3d_properties([ee_pos[2], tip[2]])

                ax.set_title(f"UVMS  —  T = {fi * dt_visual:.2f}s",
                             fontsize=11, color='#8ab4d8')
                return (hull_top, hull_bot, hull_port, hull_stbd,
                        rings_line, vp_line, fins_line, prop_line,
                        arm_line, trail_line, axis_x, axis_y, axis_z)

            ani = animation.FuncAnimation(
                fig, _update_uvms, frames=range(steps), interval=50, blit=False
            )

        # ------------------------------------------------------------------ #
        #  MODO AUV / SUBMARINO  (sem elos + ambiente água, sem braço)       #
        # ------------------------------------------------------------------ #
        elif is_free and is_new_format and is_underwater:
            ee_pts = np.array([np.array(f["links"][-1], dtype=float) for f in data])
            c      = ee_pts.mean(axis=0)
            spread = max(np.max(np.abs(ee_pts - c)) * 1.7 + 0.05, 0.3)
            ax.set_xlim(c[0]-spread, c[0]+spread)
            ax.set_ylim(c[1]-spread, c[1]+spread)
            ax.set_zlim(c[2]-spread, c[2]+spread)
            ax.set_facecolor('#0a1628')
            fig.patch.set_facecolor('#0a1628')
            ax.tick_params(colors='#8ab4d8')
            ax.xaxis.label.set_color('#8ab4d8')
            ax.yaxis.label.set_color('#8ab4d8')
            ax.zaxis.label.set_color('#8ab4d8')

            size   = spread * 0.18
            TRAIL_LEN = 120
            geom   = self._build_auv_geometry(size)

            # Paleta submarina
            C_HULL  = '#4a90c4'
            C_RING  = '#2d6a9f'
            C_FIN   = '#5ba3d9'
            C_PROP  = '#f0a500'
            C_TRAIL = '#00e5ff'
            C_AX_X  = '#ff4d4d'
            C_AX_Y  = '#66ff66'
            C_AX_Z  = '#4db8ff'

            def _make(color, lw, alpha=1.0, style='-'):
                return ax.plot([], [], [], style, lw=lw, color=color, alpha=alpha)[0]

            trail_line = _make(C_TRAIL,  1.4, 0.50)
            hull_top   = _make(C_HULL,   1.8)
            hull_bot   = _make(C_HULL,   1.8)
            hull_port  = _make(C_HULL,   1.2, 0.70)
            hull_stbd  = _make(C_HULL,   1.2, 0.70)
            rings_line = _make(C_RING,   1.0, 0.65)
            vp_line    = _make('#a8d8f0', 1.4, 0.90)
            fins_line  = _make(C_FIN,    1.6)
            prop_line  = _make(C_PROP,   1.5, 0.90)
            axis_x     = _make(C_AX_X,   2.0)
            axis_y     = _make(C_AX_Y,   2.0)
            axis_z     = _make(C_AX_Z,   2.0)

            ax.scatter(*ee_pts[0],  c='#00e676', s=60, zorder=5, label='Início',
                       depthshade=False)
            ax.scatter(*ee_pts[-1], c='#ff1744', s=60, zorder=5, marker='*',
                       label='Alvo', depthshade=False)
            ax.legend(loc='upper left', fontsize=8, facecolor='#0a1628',
                      labelcolor='#8ab4d8', edgecolor='#2d6a9f')

            trail_x, trail_y, trail_z = [], [], []

            def _update_auv(fi):
                frame = data[fi]
                pos   = np.array(frame["links"][-1], dtype=float)
                R     = np.array(frame["R"], dtype=float).reshape(3, 3)

                trail_x.append(pos[0])
                trail_y.append(pos[1])
                trail_z.append(pos[2])
                trail_line.set_data(trail_x[-TRAIL_LEN:], trail_y[-TRAIL_LEN:])
                trail_line.set_3d_properties(trail_z[-TRAIL_LEN:])

                _set_line(hull_top,   _apply_tf(geom["top"],      R, pos))
                _set_line(hull_bot,   _apply_tf(geom["bot"],      R, pos))
                _set_line(hull_port,  _apply_tf(geom["port"],     R, pos))
                _set_line(hull_stbd,  _apply_tf(geom["stbd"],     R, pos))
                _set_line(rings_line, _apply_tf(geom["rings"],    R, pos))
                _set_line(vp_line,    _apply_tf(geom["viewport"], R, pos))
                _set_line(fins_line,  _apply_tf(geom["fins"],     R, pos))
                _set_line(prop_line,  _apply_tf(geom["prop"],     R, pos))

                ax_len = size * 1.1
                for line_obj, col in [(axis_x, 0), (axis_y, 1), (axis_z, 2)]:
                    tip = pos + R[:, col] * ax_len
                    line_obj.set_data([pos[0], tip[0]], [pos[1], tip[1]])
                    line_obj.set_3d_properties([pos[2], tip[2]])

                ax.set_title(f"T = {fi * dt_visual:.2f}s", fontsize=11,
                             color='#8ab4d8')
                return (trail_line, hull_top, hull_bot, hull_port, hull_stbd,
                        rings_line, vp_line, fins_line, prop_line,
                        axis_x, axis_y, axis_z)

            ani = animation.FuncAnimation(
                fig, _update_auv, frames=range(steps), interval=50, blit=False
            )

        # ------------------------------------------------------------------ #
        #  MODO DRONE  (sem elos + ar)                                        #
        # ------------------------------------------------------------------ #
        elif is_free and is_new_format:
            # Posições do end-effector em todos os frames
            ee_pts = np.array([np.array(f["links"][-1], dtype=float) for f in data])
            c      = ee_pts.mean(axis=0)
            spread = max(np.max(np.abs(ee_pts - c)) * 1.6 + 0.05, 0.3)
            ax.set_xlim(c[0]-spread, c[0]+spread)
            ax.set_ylim(c[1]-spread, c[1]+spread)
            ax.set_zlim(c[2]-spread, c[2]+spread)

            arm_len   = spread * 0.12
            rotor_r   = arm_len * 0.45
            TRAIL_LEN = 100

            # --- objetos gráficos persistentes ---
            trail_line,  = ax.plot([], [], [], '-',  lw=1.5, color='deepskyblue',  alpha=0.55)
            body_line,   = ax.plot([], [], [], 'o-', lw=2.5, color='#2c3e50',      ms=4)
            axis_x_line, = ax.plot([], [], [], '-',  lw=2,   color='red')
            axis_y_line, = ax.plot([], [], [], '-',  lw=2,   color='limegreen')
            axis_z_line, = ax.plot([], [], [], '-',  lw=2,   color='dodgerblue')
            rotor_lines  = [ax.plot([], [], [], '-', lw=1.2, color='#7f8c8d', alpha=0.9)[0]
                            for _ in range(4)]

            # Marcadores de início e alvo
            ax.scatter(*ee_pts[0],  c='limegreen', s=70, zorder=5, label='Início')
            ax.scatter(*ee_pts[-1], c='tomato',    s=70, zorder=5, marker='*', label='Alvo')
            ax.legend(loc='upper left', fontsize=8)

            theta_r = np.linspace(0, 2 * np.pi, 24)
            trail_x, trail_y, trail_z = [], [], []

            # Direções locais dos 4 braços (em ângulos de 45 °)
            ARM_DIRS = np.array([
                [ 1,  1, 0],
                [-1,  1, 0],
                [-1, -1, 0],
                [ 1, -1, 0],
            ], dtype=float)
            ARM_DIRS /= np.linalg.norm(ARM_DIRS[0])

            def _update_drone(fi):
                frame = data[fi]
                pos   = np.array(frame["links"][-1], dtype=float)
                R     = np.array(frame["R"],         dtype=float).reshape(3, 3)

                # Trilha
                trail_x.append(pos[0]); trail_y.append(pos[1]); trail_z.append(pos[2])
                tx = trail_x[-TRAIL_LEN:]; ty = trail_y[-TRAIL_LEN:]; tz = trail_z[-TRAIL_LEN:]
                trail_line.set_data(tx, ty); trail_line.set_3d_properties(tz)

                # Corpo — dois segmentos cruzados (arm0↔arm2, arm1↔arm3)
                tips = (R @ (ARM_DIRS * arm_len).T).T + pos
                bx = [tips[0,0], pos[0], tips[2,0], np.nan,
                      tips[1,0], pos[0], tips[3,0]]
                by = [tips[0,1], pos[1], tips[2,1], np.nan,
                      tips[1,1], pos[1], tips[3,1]]
                bz = [tips[0,2], pos[2], tips[2,2], np.nan,
                      tips[1,2], pos[2], tips[3,2]]
                body_line.set_data(bx, by); body_line.set_3d_properties(bz)

                # Hélices (discos circulares no plano XY do drone)
                for k, tip in enumerate(tips):
                    circ_local  = np.array([rotor_r * np.cos(theta_r),
                                            rotor_r * np.sin(theta_r),
                                            np.zeros_like(theta_r)])
                    circ_global = (R @ circ_local).T + tip
                    rotor_lines[k].set_data(circ_global[:, 0], circ_global[:, 1])
                    rotor_lines[k].set_3d_properties(circ_global[:, 2])

                # Eixos de orientação (X=vermelho, Y=verde, Z=azul)
                for line_obj, col in [(axis_x_line, 0),
                                      (axis_y_line, 1),
                                      (axis_z_line, 2)]:
                    tip_a = pos + R[:, col] * arm_len * 1.2
                    line_obj.set_data([pos[0], tip_a[0]], [pos[1], tip_a[1]])
                    line_obj.set_3d_properties([pos[2], tip_a[2]])

                ax.set_title(f"T = {fi * dt_visual:.2f}s", fontsize=11)
                return (trail_line, body_line, axis_x_line, axis_y_line, axis_z_line,
                        *rotor_lines)

            ani = animation.FuncAnimation(
                fig, _update_drone, frames=range(steps), interval=50, blit=False
            )

        # ------------------------------------------------------------------ #
        #  MODO BRAÇO  (há elos → mostrar cadeia cinemática completa)        #
        # ------------------------------------------------------------------ #
        else:
            all_pts = []
            for frame in data:
                links = frame["links"] if is_new_format else frame
                all_pts.extend(links)
            all_pts = np.array(all_pts)
            if len(all_pts) > 0:
                mv = np.max(np.abs(all_pts)) * 1.2 + 0.1
                ax.set_xlim(-mv, mv); ax.set_ylim(-mv, mv); ax.set_zlim(-mv, mv)

            line,  = ax.plot([], [], [], 'o-', lw=3, markersize=6, color='blue')
            trace, = ax.plot([], [], [], '-',  lw=1, color='red',   alpha=0.5)
            trace_x, trace_y, trace_z = [], [], []

            def _update_arm(fi):
                frame = data[fi]
                links = frame["links"] if is_new_format else frame
                pose  = np.array(links)
                xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
                line.set_data(xs, ys); line.set_3d_properties(zs)
                trace_x.append(xs[-1]); trace_y.append(ys[-1]); trace_z.append(zs[-1])
                trace.set_data(trace_x, trace_y); trace.set_3d_properties(trace_z)
                ax.set_title(f"T = {fi * dt_visual:.2f}s")
                return line, trace

            ani = animation.FuncAnimation(
                fig, _update_arm, frames=range(steps), interval=50, blit=False
            )

        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # CAPTURA / RESTAURAÇÃO DE PARÂMETROS DA UI
    # ==========================================================================
    def _collect_ui_params(self):
        """Lê todos os widgets da aba Simulação e retorna um dict serializável."""
        return {
            # Posições e tempo
            "start":             self.entry_start.get(),
            "end":               self.entry_end.get(),
            "q_init":            self.entry_q_init.get(),
            "init_at_start":     self.init_at_start_var.get(),
            "use_last_q":        self.use_last_q_var.get(),
            "time":              self.entry_time.get(),
            "dt_physics":        self.entry_dt_physics.get(),
            "dt_visual":         self.entry_dt_visual.get(),
            # Controle
            "ctrl_mode":         self.ctrl_mode_var.get(),
            "kp":                self.entry_kp.get(),
            "zeta":              self.entry_zeta.get(),
            "ki":                self.entry_ki.get(),
            "windup":            self.entry_windup.get(),
            # ADRC
            "omega_c":           self.entry_omega_c.get(),
            "omega_o":           self.entry_omega_o.get(),
            "auto_b0":           self.auto_b0_var.get(),
            "b0":                self.entry_b0.get(),
            "gravity_ff":        self.gravity_ff_var.get(),
            "coriolis_ff":       self.coriolis_ff_var.get(),
            "z_limit":           self.entry_z_limit.get(),
            "tau_limit":         self.entry_tau_limit.get(),
            "max_wo_dt":         self.entry_max_wo_dt.get(),
            "tau_filter_alpha":  self.entry_tau_filter_alpha.get(),
            "z3_filter_alpha":   self.entry_z3_filter_alpha.get(),
            # SMC
            "smc_variant":            self.smc_variant_var.get(),
            "smc_lambda":             self.entry_lambda.get(),
            "smc_k":                  self.entry_smc_k.get(),
            "smc_phi":                self.entry_phi.get(),
            "smc_ddq_filter_alpha":   self.entry_ddq_filter_alpha.get(),
            "smc_sta_k1":             self.entry_sta_k1.get(),
            "smc_sta_k2":             self.entry_sta_k2.get(),
            # Misc
            "dq_limit":          self.entry_dq_limit.get(),
            "use_feedforward_vel": self.use_feedforward_vel_var.get(),
            "disturbance":       float(self.disturbance_slider.get()),
            # Trajetória
            "traj_type":         self.traj_type_var.get(),
            "radius":            self.entry_radius.get(),
            "normal":            self.entry_normal.get(),
            "direction":         self.switch_dir_var.get(),
            # Orientação
            "orient_mode":       self.orient_mode_var.get(),
            "slerp_roll":        self.entry_slerp_roll.get(),
            "slerp_pitch":       self.entry_slerp_pitch.get(),
            "slerp_yaw":         self.entry_slerp_yaw.get(),
            "orient_normal":     self.entry_orient_normal.get(),
            "kp_orient":         self.entry_kp_orient.get(),
            # Parâmetros físicos dinâmicos (massas, comprimentos, inércias, etc.)
            "dynamic_params":    {k: e.get() for k, e in self.dynamic_entries.items()},
            # Estado das seções expansíveis
            "adv_basic_open":    self._adv_basic_open,
            "adrc_adv_open":     self._adrc_adv_open,
        }

    def _restore_ui_params(self, params):
        """Restaura os widgets da aba Simulação a partir de um dict salvo."""
        def _set(entry, val):
            entry.delete(0, "end")
            entry.insert(0, str(val))

        if "start"             in params: _set(self.entry_start,            params["start"])
        if "end"               in params: _set(self.entry_end,              params["end"])
        if "q_init"            in params: _set(self.entry_q_init,           params["q_init"])
        if "init_at_start"     in params: self.init_at_start_var.set(       params["init_at_start"])
        if "use_last_q"        in params: self.use_last_q_var.set(          params["use_last_q"])
        if "time"              in params: _set(self.entry_time,             params["time"])
        if "dt_physics"        in params: _set(self.entry_dt_physics,       params["dt_physics"])
        if "dt_visual"         in params: _set(self.entry_dt_visual,        params["dt_visual"])

        if "ctrl_mode" in params:
            self.ctrl_mode_var.set(params["ctrl_mode"])
            self.update_ctrl_inputs(params["ctrl_mode"])

        if "kp"                in params: _set(self.entry_kp,               params["kp"])
        if "zeta"              in params: _set(self.entry_zeta,             params["zeta"])
        if "ki"                in params: _set(self.entry_ki,               params["ki"])
        if "windup"            in params: _set(self.entry_windup,           params["windup"])
        if "omega_c"           in params: _set(self.entry_omega_c,          params["omega_c"])
        if "omega_o"           in params: _set(self.entry_omega_o,          params["omega_o"])
        if "auto_b0"           in params:
            self.auto_b0_var.set(params["auto_b0"])
            self._on_auto_b0_toggle()
        if "b0"                in params: _set(self.entry_b0,               params["b0"])
        if "gravity_ff"        in params: self.gravity_ff_var.set(          params["gravity_ff"])
        if "coriolis_ff"       in params: self.coriolis_ff_var.set(         params["coriolis_ff"])
        if "z_limit"           in params: _set(self.entry_z_limit,          params["z_limit"])
        if "tau_limit"         in params: _set(self.entry_tau_limit,        params["tau_limit"])
        if "max_wo_dt"         in params: _set(self.entry_max_wo_dt,        params["max_wo_dt"])
        if "tau_filter_alpha"  in params: _set(self.entry_tau_filter_alpha, params["tau_filter_alpha"])
        if "z3_filter_alpha"   in params: _set(self.entry_z3_filter_alpha,  params["z3_filter_alpha"])
        if "smc_variant" in params:
            self.smc_variant_var.set(params["smc_variant"])
            self._update_smc_variant(params["smc_variant"])
        if "smc_lambda"           in params: _set(self.entry_lambda,              params["smc_lambda"])
        if "smc_k"                in params: _set(self.entry_smc_k,               params["smc_k"])
        if "smc_phi"              in params: _set(self.entry_phi,                 params["smc_phi"])
        if "smc_ddq_filter_alpha" in params: _set(self.entry_ddq_filter_alpha,    params["smc_ddq_filter_alpha"])
        if "smc_sta_k1"           in params: _set(self.entry_sta_k1,              params["smc_sta_k1"])
        if "smc_sta_k2"           in params: _set(self.entry_sta_k2,              params["smc_sta_k2"])
        if "dq_limit"          in params: _set(self.entry_dq_limit,         params["dq_limit"])
        if "use_feedforward_vel" in params: self.use_feedforward_vel_var.set(params["use_feedforward_vel"])
        if "disturbance" in params:
            self.disturbance_slider.set(params["disturbance"])
            self.update_disturbance_label(params["disturbance"])
        if "traj_type" in params:
            self.traj_type_var.set(params["traj_type"])
            self.update_traj_inputs(params["traj_type"])
        if "radius"    in params: _set(self.entry_radius, params["radius"])
        if "normal"    in params: _set(self.entry_normal, params["normal"])
        if "direction" in params: self.switch_dir_var.set(params["direction"])

        if "orient_mode" in params:
            self.orient_mode_var.set(params["orient_mode"])
            self._on_orient_mode_change(params["orient_mode"])
        if "slerp_roll"    in params: _set(self.entry_slerp_roll,   params["slerp_roll"])
        if "slerp_pitch"   in params: _set(self.entry_slerp_pitch,  params["slerp_pitch"])
        if "slerp_yaw"     in params: _set(self.entry_slerp_yaw,    params["slerp_yaw"])
        if "orient_normal" in params: _set(self.entry_orient_normal, params["orient_normal"])
        if "kp_orient"     in params: _set(self.entry_kp_orient,     params["kp_orient"])

        if "dynamic_params" in params:
            for k, v in params["dynamic_params"].items():
                if k in self.dynamic_entries:
                    _set(self.dynamic_entries[k], v)

        # Seções expansíveis: abre/fecha apenas se o estado salvo difere do atual
        if "adv_basic_open" in params:
            if params["adv_basic_open"] != self._adv_basic_open:
                self._toggle_adv_basic()
        if "adrc_adv_open" in params:
            if params["adrc_adv_open"] != self._adrc_adv_open:
                self._toggle_adrc_adv()

    # ==========================================================================
    # ABA 3: ANÁLISE COMPARATIVA
    # ==========================================================================
    def setup_analysis_tab(self):
        self.tab_analysis.grid_columnconfigure(0, weight=1)
        self.tab_analysis.grid_columnconfigure(1, weight=2)
        self.tab_analysis.grid_rowconfigure(0, weight=1)

        # --- Painel esquerdo: gerenciamento de sessões ---
        left = ctk.CTkFrame(self.tab_analysis)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Sessões de Comparação",
                     font=("Arial", 13, "bold")).grid(row=0, column=0, pady=(10, 5), padx=10, sticky="w")

        self._session_scroll = ctk.CTkScrollableFrame(left, label_text="Simulações adicionadas")
        self._session_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=5)
        self._session_scroll.grid_columnconfigure(0, weight=1)

        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=5)
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.btn_add_session = ctk.CTkButton(
            btn_frame, text="+ Adicionar Atual",
            fg_color="green", command=self._add_current_session
        )
        self.btn_add_session.grid(row=0, column=0, padx=2, pady=2, sticky="ew")

        self.btn_clear_sessions = ctk.CTkButton(
            btn_frame, text="Limpar Tudo",
            fg_color="firebrick", command=self._clear_sessions
        )
        self.btn_clear_sessions.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        self.btn_generate_report = ctk.CTkButton(
            left, text="GERAR RELATÓRIO 📊",
            height=40, font=ctk.CTkFont(weight="bold"),
            command=self._generate_report
        )
        self.btn_generate_report.grid(row=3, column=0, padx=8, pady=(5, 10), sticky="ew")

        # --- Painel direito: tabela de métricas resumidas ---
        right = ctk.CTkFrame(self.tab_analysis)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Métricas Resumidas",
                     font=("Arial", 13, "bold")).grid(row=0, column=0, pady=(10, 5), padx=10, sticky="w")

        self._metrics_box = ctk.CTkTextbox(right, font=("Consolas", 11), state="disabled")
        self._metrics_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._refresh_metrics_box()

    def _auto_label(self, sim_results):
        """Gera um label automático a partir dos ui_params da simulação."""
        p = sim_results.get("ui_params", {})
        ctrl = p.get("ctrl_mode", "?")
        t_total = p.get("time", "?")
        dist = p.get("disturbance", 0.0)

        if ctrl == "Torque Computado":
            kp   = p.get("kp", "?")
            zeta = p.get("zeta", "?")
            detail = f"Kp={kp} ζ={zeta}"
        elif ctrl == "ADRC (Robust)":
            wc = p.get("omega_c", "?")
            wo = p.get("omega_o", "?")
            detail = f"ωc={wc} ωo={wo}"
        elif ctrl == "Sliding Mode (SMC)":
            variant = p.get("smc_variant", "CT-SMC")
            lam = p.get("smc_lambda", "?")
            detail = f"{variant} λ={lam}"
        else:
            detail = ""

        dist_str = f" | d={dist:.1f}Nm" if dist != 0.0 else ""
        return f"{ctrl} | {detail} | T={t_total}s{dist_str}"

    def _add_current_session(self):
        if self.last_sim_results is None:
            self.log("⚠️ Rode uma simulação antes de adicionar à sessão de análise.")
            return
        label = self._auto_label(self.last_sim_results)
        session = dict(self.last_sim_results)
        session["label"] = label
        self.comparison_sessions.append(session)
        self._rebuild_session_list()
        self._refresh_metrics_box()
        self.log(f"✅ Simulação adicionada à análise: {label}")

    def _rebuild_session_list(self):
        for widget in self._session_scroll.winfo_children():
            widget.destroy()
        self._session_rows.clear()

        for idx, sess in enumerate(self.comparison_sessions):
            row = ctk.CTkFrame(self._session_scroll, border_width=1)
            row.pack(fill="x", pady=2, padx=2)
            row.grid_columnconfigure(0, weight=1)

            lbl_var = ctk.StringVar(value=sess["label"])
            entry = ctk.CTkEntry(row, textvariable=lbl_var, font=("Consolas", 10))
            entry.grid(row=0, column=0, sticky="ew", padx=(6, 2), pady=4)

            def _on_label_change(var=lbl_var, i=idx):
                self.comparison_sessions[i]["label"] = var.get()

            lbl_var.trace_add("write", lambda *_, var=lbl_var, i=idx: _on_label_change(var, i))

            elapsed = sess.get("elapsed_time")
            elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "—"
            n_dof = sess["err"].shape[1] if sess["err"].ndim > 1 else 1
            info = ctk.CTkLabel(row, text=f"DOF={n_dof} | cômputo={elapsed_str}",
                                font=("Consolas", 9), text_color="gray70")
            info.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 3))

            def _remove(i=idx):
                self.comparison_sessions.pop(i)
                self._rebuild_session_list()
                self._refresh_metrics_box()

            btn_del = ctk.CTkButton(row, text="✕", width=28, height=28,
                                    fg_color="firebrick", command=_remove)
            btn_del.grid(row=0, column=1, rowspan=2, padx=(2, 6), pady=4)

            self._session_rows.append(row)

    def _clear_sessions(self):
        self.comparison_sessions.clear()
        self._rebuild_session_list()
        self._refresh_metrics_box()

    def _refresh_metrics_box(self):
        self._metrics_box.configure(state="normal")
        self._metrics_box.delete("1.0", "end")
        if not self.comparison_sessions:
            self._metrics_box.insert("end",
                "Nenhuma sessão adicionada.\n\n"
                "Rode uma simulação e clique em\n"
                "\"+ Adicionar Atual\" para começar.")
            self._metrics_box.configure(state="disabled")
            return

        col_w = 14
        n_dof = self.comparison_sessions[0]["err"].shape[1] \
            if self.comparison_sessions[0]["err"].ndim > 1 else 1

        header_parts = ["Sessão".ljust(30)]
        for d in range(n_dof):
            header_parts.append(f"RMSE J{d+1}".rjust(col_w))
        header_parts += [
            "MaxErr".rjust(col_w),
            "Energia".rjust(col_w),
            "MaxTau".rjust(col_w),
            "Chat.".rjust(col_w),
            "Tempo(s)".rjust(col_w),
        ]
        header = "".join(header_parts)
        sep = "─" * len(header)

        self._metrics_box.insert("end", header + "\n" + sep + "\n")

        for sess in self.comparison_sessions:
            m = self._compute_metrics(sess["t"], sess["err"], sess["tau"])
            label = sess["label"][:29].ljust(30)
            row_parts = [label]
            for d in range(n_dof):
                row_parts.append(f"{m['rmse'][d]:.4f}".rjust(col_w))
            row_parts += [
                f"{m['max_err']:.4f}".rjust(col_w),
                f"{m['energy']:.2f}".rjust(col_w),
                f"{m['max_tau']:.2f}".rjust(col_w),
                f"{m['chattering']:.4f}".rjust(col_w),
                f"{sess.get('elapsed_time', 0.0):.2f}".rjust(col_w),
            ]
            self._metrics_box.insert("end", "".join(row_parts) + "\n")

        self._metrics_box.configure(state="disabled")

    # ==========================================================================
    # MÉTRICAS
    # ==========================================================================
    @staticmethod
    def _compute_metrics(t, err, tau):
        """Calcula métricas de qualidade de controle.

        Parameters
        ----------
        t   : (N,) array de tempo
        err : (N, DOF) array de erros de junta
        tau : (N, DOF) array de torques aplicados

        Returns
        -------
        dict com rmse (por junta), max_err (global), final_err (global),
        energy (global), max_tau (global), chattering (global)
        """
        if err.ndim == 1:
            err = err[:, np.newaxis]
        if tau.ndim == 1:
            tau = tau[:, np.newaxis]

        dt = np.diff(t)
        if len(dt) == 0:
            dt = np.array([1.0])

        # Por junta
        rmse = np.sqrt(np.mean(err ** 2, axis=0))

        # Global (sobre todas as juntas)
        max_err = float(np.max(np.abs(err)))

        # Regime permanente: média dos últimos 5% dos pontos
        tail = max(1, int(len(t) * 0.05))
        final_err = float(np.mean(np.abs(err[-tail:, :])))

        # Energia: integral de |τ|² no tempo
        tau_sq = tau[:-1, :] ** 2
        energy = float(np.sum(tau_sq * dt[:, np.newaxis]))

        max_tau = float(np.max(np.abs(tau)))

        # Chattering: variação média do torque
        dtau = np.abs(np.diff(tau, axis=0))
        chattering = float(np.mean(dtau))

        return {
            "rmse":       rmse,
            "max_err":    max_err,
            "final_err":  final_err,
            "energy":     energy,
            "max_tau":    max_tau,
            "chattering": chattering,
        }

    # ==========================================================================
    # RELATÓRIO
    # ==========================================================================
    def _generate_report(self):
        if not self.comparison_sessions:
            self.log("⚠️ Adicione ao menos uma simulação na aba Análise antes de gerar o relatório.")
            return

        sessions = self.comparison_sessions
        n_sess = len(sessions)
        n_dof = sessions[0]["err"].shape[1] if sessions[0]["err"].ndim > 1 else 1

        COLORS = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]

        def _color(i):
            return COLORS[i % len(COLORS)]

        # ------------------------------------------------------------------
        # Layout: 3 linhas
        #   Linha 0: erro de junta (n_dof colunas)
        #   Linha 1: torque de junta (n_dof colunas)
        #   Linha 2: 4 gráficos de barras (RMSE médio, Energia, Max τ, Tempo)
        # ------------------------------------------------------------------
        n_bar_cols = 4
        n_cols = max(n_dof, n_bar_cols)

        fig = plt.figure("Relatório Comparativo — Hephaestus", figsize=(4 * n_cols, 12))
        fig.patch.set_facecolor("#1a1a2e")

        # Grade manual: 3 linhas × n_cols colunas
        gs_top  = fig.add_gridspec(1, n_dof,   top=0.96, bottom=0.70, hspace=0.35, wspace=0.35)
        gs_mid  = fig.add_gridspec(1, n_dof,   top=0.65, bottom=0.39, hspace=0.35, wspace=0.35)
        gs_bot  = fig.add_gridspec(1, n_bar_cols, top=0.34, bottom=0.08, hspace=0.35, wspace=0.40)

        _ax_style = dict(facecolor="#16213e")

        def _style_ax(ax, title, ylabel, xlabel="Tempo (s)"):
            ax.set_facecolor("#16213e")
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
            ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.grid(True, color="#2a2a4a", linewidth=0.6)

        # --- Linha 0: Erro por junta ---
        axes_err = [fig.add_subplot(gs_top[0, d]) for d in range(n_dof)]
        for d, ax in enumerate(axes_err):
            for i, sess in enumerate(sessions):
                err_d = sess["err"][:, d] if sess["err"].ndim > 1 else sess["err"]
                ax.plot(sess["t"], err_d, color=_color(i),
                        lw=1.2, alpha=0.85, label=sess["label"])
            _style_ax(ax, f"Erro — Junta {d+1} (rad/m)", "Erro")
            if d == n_dof - 1:
                ax.legend(fontsize=7, loc="upper right",
                          facecolor="#0f0f23", labelcolor="white",
                          edgecolor="#444466", framealpha=0.8)

        # --- Linha 1: Torque por junta ---
        axes_tau = [fig.add_subplot(gs_mid[0, d]) for d in range(n_dof)]
        for d, ax in enumerate(axes_tau):
            for i, sess in enumerate(sessions):
                tau_d = sess["tau"][:, d] if sess["tau"].ndim > 1 else sess["tau"]
                ax.plot(sess["t"], tau_d, color=_color(i),
                        lw=1.2, alpha=0.85, label=sess["label"])
            _style_ax(ax, f"Torque — Junta {d+1} (Nm)", "Torque (Nm)")

        # --- Linha 2: Barras comparativas ---
        labels_bar = [s["label"] for s in sessions]
        x = np.arange(n_sess)
        bar_w = max(0.15, min(0.5, 0.6 / n_sess))

        metrics_list = [self._compute_metrics(s["t"], s["err"], s["tau"]) for s in sessions]

        bar_specs = [
            (fig.add_subplot(gs_bot[0, 0]),
             [float(np.mean(m["rmse"])) for m in metrics_list],
             "RMSE médio (rad/m)", "RMSE"),
            (fig.add_subplot(gs_bot[0, 1]),
             [m["energy"] for m in metrics_list],
             "Energia (∫|τ|²dt)", "Energia"),
            (fig.add_subplot(gs_bot[0, 2]),
             [m["max_tau"] for m in metrics_list],
             "Torque máximo (Nm)", "Max |τ|"),
            (fig.add_subplot(gs_bot[0, 3]),
             [s.get("elapsed_time", 0.0) or 0.0 for s in sessions],
             "Tempo de cômputo (s)", "Tempo (s)"),
        ]

        for ax_b, values, title, ylabel in bar_specs:
            bars = ax_b.bar(x, values, width=bar_w * n_sess,
                            color=[_color(i) for i in range(n_sess)],
                            edgecolor="#1a1a2e", linewidth=0.8)
            for bar_obj, val in zip(bars, values):
                ax_b.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                          bar_obj.get_height() * 1.02,
                          f"{val:.3g}", ha="center", va="bottom",
                          color="white", fontsize=7)
            ax_b.set_xticks(x)
            ax_b.set_xticklabels(
                [lb[:18] for lb in labels_bar],
                rotation=25, ha="right", fontsize=7
            )
            _style_ax(ax_b, title, ylabel, xlabel="")

        # Título principal
        fig.text(0.5, 0.99,
                 "Relatório Comparativo de Controladores — Hephaestus",
                 ha="center", va="top", fontsize=13, color="white", fontweight="bold")

        # Rodapé com métricas extras (chattering, erro final)
        footer_lines = []
        for i, (sess, m) in enumerate(zip(sessions, metrics_list)):
            elapsed = sess.get("elapsed_time") or 0.0
            footer_lines.append(
                f"[{_color(i)} ■]  {sess['label'][:50]:<50}  "
                f"ErroFinal={m['final_err']:.4f}  "
                f"Chat={m['chattering']:.4f}  "
                f"Cômputo={elapsed:.2f}s"
            )
        fig.text(0.01, 0.04, "\n".join(footer_lines),
                 fontsize=7, color="#aaaaaa",
                 fontfamily="monospace", va="top")

        # Botão de exportar PDF dentro da janela matplotlib
        ax_save = fig.add_axes([0.87, 0.005, 0.12, 0.025])
        ax_save.set_axis_off()
        from matplotlib.widgets import Button as MplButton
        btn_save = MplButton(ax_save, "Exportar PDF", color="#2a2a4a", hovercolor="#444466")

        def _on_save(_event):
            from tkinter import filedialog as _fd
            path = _fd.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"), ("Todos", "*.*")],
                title="Exportar Relatório",
            )
            if path:
                fig.savefig(path, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                self.log(f"✅ Relatório exportado: {path}")

        btn_save.on_clicked(_on_save)

        plt.show()

    # ==========================================================================
    # MENU DE ARQUIVO
    # ==========================================================================
    def _create_menu(self):
        menubar = tk.Menu(self)
        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Arquivo", menu=self.file_menu)
        self.file_menu.add_command(label="Salvar Modelo...",     command=self.save_model)
        self.file_menu.add_command(label="Carregar Modelo...",   command=self.load_model)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Salvar Simulação...",  command=self.save_simulation)
        self.file_menu.add_command(label="Carregar Simulação...", command=self.load_simulation)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Sair",                 command=self.on_closing)
        self.config(menu=menubar)

    def _update_menu_state(self):
        model_state = "normal" if self.active_sim is not None else "disabled"
        sim_state   = "normal" if self.last_sim_results is not None else "disabled"
        self.file_menu.entryconfig(0, state=model_state)   # Salvar Modelo
        self.file_menu.entryconfig(3, state=sim_state)     # Salvar Simulação

    # ==========================================================================
    # SALVAR / CARREGAR MODELO (.hmodel)
    # ==========================================================================
    def save_model(self):
        if not self.active_bot:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".hmodel",
            filetypes=[("Modelo Hephaestus", "*.hmodel"), ("Todos os arquivos", "*.*")],
            title="Salvar Modelo"
        )
        if not filepath:
            return
        try:
            data = {
                "type":  "hmodel",
                "bot":   self.active_bot,
                "mode":  self.mode_var.get(),
            }
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.log(f"✅ Modelo salvo em: {filepath}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar modelo", str(e))
            self.log(f"❌ Erro ao salvar modelo: {e}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Modelo Hephaestus", "*.hmodel"), ("Todos os arquivos", "*.*")],
            title="Carregar Modelo"
        )
        if not filepath:
            return
        self.log(f"Carregando modelo de: {filepath} ...")
        threading.Thread(target=self._load_model_thread, args=(filepath, False), daemon=True).start()

    def _load_model_thread(self, filepath, has_sim_results):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            bot  = data["bot"]
            mode = data["mode"]

            if self.active_sim is not None:
                self.active_sim.close()

            self.active_bot = bot
            self.log("Compilando equações (lambdify)...")
            sim_mode = "Hydro" if mode == "Água (UVMS)" else "Air"
            self.active_sim = RobotSimulator(self.active_bot, mode=sim_mode)

            sim_results = data.get("sim_results") if has_sim_results else None
            self.after(0, lambda: self._finish_load_model(mode, sim_results))

        except Exception as e:
            self.log(f"❌ Erro ao carregar arquivo: {e}")
            import traceback
            traceback.print_exc()

    def _finish_load_model(self, mode, sim_results=None):
        self.mode_var.set(mode)
        self.update_mode_color(mode)

        for row_data in self.joint_rows:
            row_data["frame"].destroy()
        self.joint_rows.clear()

        bot = self.active_bot
        for j_type, l_vec in zip(bot.joint_config, bot.link_vectors_mask):
            self.add_joint()
            row = self.joint_rows[-1]
            row["dd"].set(j_type)
            if int(l_vec[0]): row["cx"].select()
            else:              row["cx"].deselect()
            if int(l_vec[1]): row["cy"].select()
            else:              row["cy"].deselect()
            if int(l_vec[2]): row["cz"].select()
            else:              row["cz"].deselect()

        self.last_sim_results = None
        self.generate_sim_inputs()
        self.toggle_sim_tab(True)
        self.tabview.set("Simulação")

        if sim_results is not None:
            t         = sim_results["t"]
            err       = sim_results["err"]
            tau       = sim_results["tau"]
            anim_data = sim_results["anim_data"]
            dt_visual = sim_results["dt_visual"]
            self.last_sim_results = sim_results
            self.last_anim_data   = anim_data
            self.last_dt_visual   = dt_visual
            if "ui_params" in sim_results:
                self._restore_ui_params(sim_results["ui_params"])
            self.plot_results(t, err, tau)
            self.btn_anim3d.configure(state="normal")
            self.log(f"✅ Simulação restaurada ({len(t)} pontos).")
        else:
            self.btn_anim3d.configure(state="disabled")

        self._update_menu_state()
        self.log(f"✅ Modelo carregado com sucesso! ({len(bot.joint_config)} juntas)")

    # ==========================================================================
    # SALVAR / CARREGAR SIMULAÇÃO (.hsim)
    # ==========================================================================
    def save_simulation(self):
        if not self.active_bot or self.last_sim_results is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".hsim",
            filetypes=[("Simulação Hephaestus", "*.hsim"), ("Todos os arquivos", "*.*")],
            title="Salvar Simulação"
        )
        if not filepath:
            return
        try:
            data = {
                "type":        "hsim",
                "bot":         self.active_bot,
                "mode":        self.mode_var.get(),
                "sim_results": self.last_sim_results,
            }
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.log(f"✅ Simulação salva em: {filepath}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar simulação", str(e))
            self.log(f"❌ Erro ao salvar simulação: {e}")

    def load_simulation(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Simulação Hephaestus", "*.hsim"),
                       ("Modelo Hephaestus",    "*.hmodel"),
                       ("Todos os arquivos",    "*.*")],
            title="Carregar Simulação"
        )
        if not filepath:
            return
        self.log(f"Carregando simulação de: {filepath} ...")
        threading.Thread(target=self._load_model_thread, args=(filepath, True), daemon=True).start()

if __name__ == "__main__":
    # Obrigatório para PyInstaller + multiprocessing no Windows.
    # Sem isso, o ProcessPoolExecutor no cálculo de Coriolis causa spawn de
    # processos que reabrem o executável e geram o erro "A process in the
    # process pool was terminated abruptly".
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
