"""
benchmark_parallel.py
=====================
Avalia o desempenho do SUMÉ em duas dimensões:

  1. Speedup de modelagem (Coriolis + dM/dq) em função do número de
     workers, para robôs de 3, 4, 6 e 9 GDL.

  2. Speedup de simulação numérica (avaliação paralela de M, C, G)
     comparando execução sequencial vs. paralela para um robô de 6 GDL.

Os resultados são salvos em ``benchmark_results.csv`` e dois gráficos
PNG são gerados prontos para inclusão no TCC.

Uso
---
    python benchmark_parallel.py

Nota
----
O script pode levar 30–90 minutos dependendo do hardware e do número
de GDL testados. Cada configuração é executada uma única vez (sem
repetição estatística) para manter o tempo total razoável.
A estimativa de progresso é impressa antes de cada medição.
"""

import csv
import multiprocessing
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")   # backend não-interativo (sem janela)
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _silent_log(*args, **kwargs):   # noqa: D401
    """Logger silencioso para suprimir saída da engine durante o benchmark."""


def _build_params(n_dof, L=0.1, mass=1.0, g=9.81):
    """Gera dicionário de parâmetros físicos para um robô serial simples.

    Cada elo tem:
      - massa ``mass``, comprimento ``L``;
      - CM no ponto médio do elo (``cz = L/2``);
      - tensores de inércia de um cilindro uniforme.
    """
    params = {"g": g}
    for i in range(1, n_dof + 1):
        params[f"m{i}"]    = mass
        params[f"L{i}"]    = L
        params[f"cx{i}"]   = 0.0
        params[f"cy{i}"]   = 0.0
        params[f"cz{i}"]   = L / 2.0
        params[f"Ixx{i}"]  = mass * L ** 2 / 12.0
        params[f"Iyy{i}"]  = mass * L ** 2 / 12.0
        params[f"Izz{i}"]  = 1.0e-4
    return params


# ---------------------------------------------------------------------------
# Parte 1 – Benchmark de Modelagem (Coriolis)
# ---------------------------------------------------------------------------

def _model_robot(n_dof, workers):
    """Instancia e modela um robô serial com ``n_dof`` juntas Rz.

    Retorna (t_kine, t_mg, t_coriolis, engine_object).
    """
    from engine import RobotMathEngine

    joint_config  = ["Rz"] * n_dof
    link_vectors  = [[0, 0, 1]] * n_dof

    bot = RobotMathEngine(
        joint_config,
        link_vectors,
        logger_callback=_silent_log,
        num_workers=workers,
    )

    t0 = time.perf_counter(); bot.step_1_kinematics();        t_kine     = time.perf_counter() - t0
    t0 = time.perf_counter(); bot.step_2_jacobian_M_G();      t_mg       = time.perf_counter() - t0
    t0 = time.perf_counter(); bot.step_3_coriolis_combined(); t_coriolis = time.perf_counter() - t0
    bot.step_4_prepare_export()

    return t_kine, t_mg, t_coriolis, bot


def benchmark_modeling(dof_list, worker_options, out_rows):
    """Mede tempo de modelagem (Coriolis) por GDL e número de workers.

    Para cada GDL, o tempo com 1 worker (serial) serve de referência
    para o cálculo do speedup dos demais.
    """
    print("\n=== PARTE 1: Speedup de Modelagem ===")
    for n in dof_list:
        t_serial_coriolis = None
        for w in worker_options:
            print(f"  [{n} GDL | {w:2d} worker(s)] modelando...", flush=True)
            t_k, t_mg, t_c, _ = _model_robot(n, w)
            total = t_k + t_mg + t_c

            if w == 1:
                t_serial_coriolis = t_c if t_c > 0 else 1e-9
            speedup = t_serial_coriolis / t_c if (t_serial_coriolis and t_c > 0) else 1.0

            out_rows.append({
                "part":               "modeling",
                "n_dof":              n,
                "workers":            w,
                "t_kine_s":           round(t_k,    3),
                "t_mg_s":             round(t_mg,   3),
                "t_coriolis_s":       round(t_c,    3),
                "t_total_s":          round(total,  3),
                "speedup_coriolis":   round(speedup, 3),
                "use_parallel":       "",
                "elapsed_sim_s":      "",
                "speedup_sim":        "",
            })
            print(f"    Coriolis={t_c:.1f}s | Total={total:.1f}s | Speedup={speedup:.2f}×")
    return out_rows


# ---------------------------------------------------------------------------
# Parte 2 – Benchmark de Simulação (paralelo vs. sequencial)
# ---------------------------------------------------------------------------

def benchmark_simulation(n_dof, max_workers_sim, out_rows):
    """Mede tempo de simulação sequential vs. paralelo para ``n_dof`` GDL.

    A compilação (lambdify) é executada uma única vez e o modelo é
    reutilizado nas duas configurações.
    """
    from engine import RobotMathEngine
    from simulator import RobotSimulator

    cpu_count = os.cpu_count() or 1
    workers_for_model = min(cpu_count, 6)

    print(f"\n=== PARTE 2: Speedup de Simulação ({n_dof} GDL) ===")
    print(f"  Modelando com {workers_for_model} worker(s)...", flush=True)
    _, _, _, bot = _model_robot(n_dof, workers_for_model)

    print("  Compilando model (lambdify)...", flush=True)
    t0 = time.perf_counter()
    sim = RobotSimulator(bot, mode="Air")
    t_compile = time.perf_counter() - t0
    print(f"  Compilação concluída em {t_compile:.1f}s")

    params = _build_params(n_dof)
    sim.set_parameters(params)

    L   = 0.1
    Pi  = [0.0, 0.0, n_dof * L]   # efetuador na posição home (todos q=0)
    Pf  = Pi.copy()                 # sem deslocamento: mede overhead puro de avaliação

    t_seq = None
    configs = [(False, 1), (True, min(max_workers_sim, cpu_count))]
    for use_par, n_workers in configs:
        label = "sequencial" if not use_par else f"paralelo ({n_workers} workers)"
        print(f"  Simulando {label}...", flush=True)
        try:
            _, _, _, _, elapsed = sim.run(
                t_total=3.0,
                Pi_list=Pi,
                Pf_list=Pf,
                Kp_val=10.0,
                traj_mode="Line",
                dt_physics=0.005,
                dt_visual=0.05,
                use_parallel=use_par,
                max_workers=n_workers,
            )
        except Exception as exc:
            print(f"    ERRO: {exc}")
            elapsed = float("nan")

        if not use_par:
            t_seq = elapsed if elapsed > 0 else 1e-9
        speedup_sim = t_seq / elapsed if (t_seq and elapsed > 0) else 1.0

        out_rows.append({
            "part":               "simulation",
            "n_dof":              n_dof,
            "workers":            n_workers,
            "t_kine_s":           "",
            "t_mg_s":             "",
            "t_coriolis_s":       "",
            "t_total_s":          "",
            "speedup_coriolis":   "",
            "use_parallel":       use_par,
            "elapsed_sim_s":      round(elapsed, 3),
            "speedup_sim":        round(speedup_sim, 3),
        })
        print(f"    elapsed={elapsed:.1f}s | speedup={speedup_sim:.2f}×")

    sim.close()
    return out_rows


# ---------------------------------------------------------------------------
# Saída: CSV + gráficos
# ---------------------------------------------------------------------------

def save_csv(rows, path="benchmark_results.csv"):
    if not rows:
        print("Nenhum resultado para salvar.")
        return
    keys = [
        "part", "n_dof", "workers", "t_kine_s", "t_mg_s",
        "t_coriolis_s", "t_total_s", "speedup_coriolis",
        "use_parallel", "elapsed_sim_s", "speedup_sim",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados salvos em {path}")


def plot_results(rows, out_dir="."):
    """Gera dois gráficos PNG prontos para o TCC."""
    modeling   = [r for r in rows if r.get("part") == "modeling"]
    simulation = [r for r in rows if r.get("part") == "simulation"]

    # ---- Figura 1: Tempo de Coriolis e Speedup por GDL ----
    if modeling:
        dof_vals    = sorted({r["n_dof"] for r in modeling})
        worker_vals = sorted({r["workers"] for r in modeling})

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        markers = ["o", "s", "^", "D", "v"]
        colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Painel esquerdo: tempo absoluto
        ax = axes[0]
        for idx, w in enumerate(worker_vals):
            xs = [r["n_dof"]       for r in modeling if r["workers"] == w]
            ys = [r["t_coriolis_s"] for r in modeling if r["workers"] == w]
            lbl = f"{w} worker{'s' if w > 1 else ''}"
            ax.plot(xs, ys,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    label=lbl, linewidth=2, markersize=8)
        ax.set_xlabel("Graus de Liberdade (GDL)", fontsize=12)
        ax.set_ylabel("Tempo (s)", fontsize=12)
        ax.set_title("Tempo de Cálculo de Coriolis\nSequencial vs. Paralelo", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.set_xticks(dof_vals)

        # Painel direito: speedup
        ax2 = axes[1]
        for idx, w in enumerate(worker_vals):
            if w == 1:
                continue
            xs = [r["n_dof"]           for r in modeling if r["workers"] == w]
            ys = [r["speedup_coriolis"] for r in modeling if r["workers"] == w]
            ax2.plot(xs, ys,
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     label=f"{w} workers", linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (serial)")
        ax2.set_xlabel("Graus de Liberdade (GDL)", fontsize=12)
        ax2.set_ylabel("Speedup (×)", fontsize=12)
        ax2.set_title("Speedup do Cálculo de Coriolis\nem Função do GDL", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.4)
        ax2.set_xticks(dof_vals)

        plt.tight_layout()
        path1 = os.path.join(out_dir, "benchmark_modeling.png")
        plt.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfico salvo em {path1}")

    # ---- Figura 2: Tempo de Simulação Seq. vs Par. ----
    if simulation:
        n_dof_sim = simulation[0]["n_dof"]
        labels    = ["Sequencial", "Paralelo"]
        times     = [r["elapsed_sim_s"] for r in simulation]

        fig, ax = plt.subplots(figsize=(6, 5))
        bar_colors = ["#1f77b4", "#2ca02c"]
        bars = ax.bar(labels, times, color=bar_colors, width=0.45, edgecolor="black")
        for bar, val in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.02,
                f"{val:.1f} s",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
        speedup_val = simulation[1]["speedup_sim"] if len(simulation) > 1 else 1.0
        ax.set_ylabel("Tempo de Execução (s)", fontsize=12)
        ax.set_title(
            f"Simulação Sequencial vs. Paralela\n"
            f"({n_dof_sim} GDL, 3 s simulados)  |  Speedup = {speedup_val:.2f}×",
            fontsize=12,
        )
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_ylim(0, max(times) * 1.30)
        plt.tight_layout()
        path2 = os.path.join(out_dir, "benchmark_simulation.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfico salvo em {path2}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()   # obrigatório no Windows com PyInstaller

    cpu_count = os.cpu_count() or 1
    print(f"Sistema: {cpu_count} CPUs lógicas detectadas.")
    print("Este benchmark pode levar 30–90 minutos. Pressione Ctrl+C para interromper.\n")

    # ------------------------------------------------------------------
    # Configuração dos testes
    # ------------------------------------------------------------------
    # GDL testados na modelagem (12 GDL omitido: custo simbólico excessivo
    # para repetição estatística — use os dados do experimento principal).
    DOF_MODELING   = [3, 4, 6, 9]

    # Workers: serial (1) + incrementos até cpu_count
    WORKER_OPTIONS = sorted({1, 2, min(4, cpu_count), min(cpu_count, 6)})

    # GDL para o teste de simulação (6 GDL: modelo de compilação razoável)
    DOF_SIM            = 6
    MAX_WORKERS_SIM    = min(3, cpu_count)    # 3 workers: um por função (M, C, G)

    out_rows = []

    # ------------------------------------------------------------------
    # Parte 1: Modelagem
    # ------------------------------------------------------------------
    try:
        benchmark_modeling(DOF_MODELING, WORKER_OPTIONS, out_rows)
    except KeyboardInterrupt:
        print("\n[Parte 1 interrompida pelo usuário]")

    # ------------------------------------------------------------------
    # Parte 2: Simulação
    # ------------------------------------------------------------------
    try:
        benchmark_simulation(DOF_SIM, MAX_WORKERS_SIM, out_rows)
    except KeyboardInterrupt:
        print("\n[Parte 2 interrompida pelo usuário]")

    # ------------------------------------------------------------------
    # Salva resultados e gera gráficos
    # ------------------------------------------------------------------
    save_csv(out_rows)
    plot_results(out_rows)

    print("\n=== Benchmark concluído. ===")
    print("Arquivos gerados:")
    print("  benchmark_results.csv    — dados brutos para tabela no TCC")
    print("  benchmark_modeling.png   — Figura: speedup de modelagem")
    print("  benchmark_simulation.png — Figura: speedup de simulação")
