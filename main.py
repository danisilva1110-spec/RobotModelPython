import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.octave import octave_code
import os
import pickle
import threading
import sys
import numpy as np
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
        self.after(10, lambda: self.state("zoomed"))
        
        # Encerramento seguro
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variáveis de Estado
        self.active_bot = None       
        self.active_sim = None       
        self.joint_rows = []
        self.last_sim_results = None  # (t, err, tau, anim_data, dt_visual)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- MENU ---
        self._create_menu()

        # --- ABAS ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.tab_model = self.tabview.add("Modelagem")
        self.tab_sim = self.tabview.add("Simulação")
        
        self.setup_modeling_tab()
        self.setup_simulation_tab()
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
            
            if modo == "Água (UVMS)":
                self.active_bot = RobotMathHydro(j_types, l_vecs, logger_callback=self.log)
            else:
                self.active_bot = RobotMathEngine(j_types, l_vecs, logger_callback=self.log)

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
        ctk.CTkCheckBox(
            self.adrc_frame, text="FF de gravidade  G(q)",
            variable=self.gravity_ff_var,
        ).pack(anchor="w", pady=(0, 3))

        self.coriolis_ff_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self.adrc_frame, text="FF de Coriolis  C(q,dq)",
            variable=self.coriolis_ff_var,
        ).pack(anchor="w", pady=(0, 3))

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

            ctrl_params = {"type": ctrl_mode}
            if ctrl_mode == "ADRC (Robust)":
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

        dist_value = float(self.disturbance_slider.get())

        ctrl_label = ctrl_params.get("type", ctrl_mode)
        self.log(
            "Iniciando Simulação "
            f"(Modo: {mode_str}, Controle: {ctrl_label}, Perturbação: {dist_value:.1f} Nm)..."
        )
        
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
                disturbance_torque=dist_value
            )
            
            self.last_anim_data = anim_data
            self.last_dt_visual = getattr(self.active_sim, "last_dt_visual", dt_visual)
            self.last_sim_results = (t, err, tau, anim_data, self.last_dt_visual)
            self.plot_results(t, err, tau)
            self._update_menu_state()
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
            t, err, tau, anim_data, dt_visual = sim_results
            self.last_sim_results  = sim_results
            self.last_anim_data    = anim_data
            self.last_dt_visual    = dt_visual
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
    app = App()
    app.mainloop()
