import customtkinter as ctk
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import modules
from engine import RobotMathEngine, RobotMathHydro
from simulator import RobotSimulator

ctk.set_appearance_mode("Dark")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hephaestus v4.0 - Integrated Environment")
        self.geometry("1200x800")
        
        self.active_bot = None
        self.active_sim = None
        
        # Tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_model = self.tabview.add("Modeling")
        self.tab_sim = self.tabview.add("Simulation")
        
        self.build_modeling_tab()
        self.build_simulation_tab()
        
        # Lock simulation until modeling is done or loaded
        self.toggle_sim_tab(False)

    def toggle_sim_tab(self, enable):
        state = "normal" if enable else "disabled"
        # CTk doesn't support disabling tabs directly easily, 
        # so we can just switch back if user clicks or hide content.
        # For now, we assume user flow.

    # --- MODELING TAB ---
    def build_modeling_tab(self):
        # (Your existing setup for joints, links, etc.)
        btn = ctk.CTkButton(self.tab_model, text="GENERATE MODEL", command=self.run_modeling)
        btn.pack(pady=20)
        
        btn_save = ctk.CTkButton(self.tab_model, text="Save JSON", command=self.save_json)
        btn_save.pack(pady=5)
        btn_load = ctk.CTkButton(self.tab_model, text="Load JSON", command=self.load_json)
        btn_load.pack(pady=5)

    def run_modeling(self):
        # ... Instantiate RobotMathHydro or Engine ...
        # self.active_bot = RobotMathHydro(...)
        # self.active_bot.run_full_process()
        
        # Enable Simulation
        self.active_sim = RobotSimulator(self.active_bot, mode="Hydro")
        self.refresh_sim_inputs()
        self.tabview.set("Simulation")

    # --- SIMULATION TAB ---
    def build_simulation_tab(self):
        self.frame_inputs = ctk.CTkScrollableFrame(self.tab_sim, width=300, label_text="Parameters")
        self.frame_inputs.pack(side="left", fill="y", padx=10, pady=10)
        
        self.frame_plots = ctk.CTkFrame(self.tab_sim)
        self.frame_plots.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Simulation Controls
        ctk.CTkLabel(self.frame_inputs, text="Start Pos (x,y,z)").pack()
        self.entry_Pi = ctk.CTkEntry(self.frame_inputs)
        self.entry_Pi.pack()
        
        ctk.CTkLabel(self.frame_inputs, text="End Pos (x,y,z)").pack()
        self.entry_Pf = ctk.CTkEntry(self.frame_inputs)
        self.entry_Pf.pack()
        
        ctk.CTkLabel(self.frame_inputs, text="Gain Kp").pack()
        self.entry_Kp = ctk.CTkEntry(self.frame_inputs)
        self.entry_Kp.insert(0, "100")
        self.entry_Kp.pack()

        # Dynamic Parameters Container
        self.lbl_params = ctk.CTkLabel(self.frame_inputs, text="Physical Constants")
        self.lbl_params.pack(pady=(20,5))
        self.dynamic_inputs = {} # Store entry widgets here
        
        btn_run = ctk.CTkButton(self.frame_inputs, text="RUN SIMULATION", fg_color="green", command=self.run_sim)
        btn_run.pack(pady=20)

    def refresh_sim_inputs(self):
        # Clear old
        for w in self.dynamic_inputs.values(): w.destroy()
        self.dynamic_inputs = {}
        
        # Generate inputs for m1, L1, Ixx1, rho, etc.
        # Retrieved from self.active_sim.sym_vars
        # (Implementation details similar to previous response)
        pass

    def run_sim(self):
        # 1. Collect inputs
        # 2. Call self.active_sim.set_parameters()
        # 3. Run sim
        # t, err, tau = self.active_sim.run(...)
        # 4. Plot results
        self.plot_results(t, err, tau)

    def plot_results(self, t, err, tau):
        # Matplotlib logic
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t, err)
        ax1.set_title("Joint Errors")
        ax2.plot(t, tau)
        ax2.set_title("Torques")
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_plots)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- JSON HANDLING ---
    def save_json(self):
        # Serialize self.active_bot config and parameters
        pass

    def load_json(self):
        # Load config, instantiate bot, populate inputs
        pass

if __name__ == "__main__":
    app = App()
    app.mainloop()