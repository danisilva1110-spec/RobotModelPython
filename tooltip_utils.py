"""
tooltip_utils.py
Rich tooltip system for the Hephaestus simulation tab.

Renders LaTeX-like formulas via matplotlib mathtext (no external LaTeX install
needed), embeds them as tkinter PhotoImages inside a dark Toplevel popup.

Block types accepted in the blocks list:
  {"type": "title",  "content": str}   – bold header
  {"type": "text",   "content": str}   – plain description
  {"type": "math",   "formula": str}   – formula rendered via mathtext
  {"type": "ref",    "content": str}   – citation in italic style
  {"type": "sep"}                      – thin horizontal separator
"""

import io
import base64
import tkinter as tk
import matplotlib.figure as mfig
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---------------------------------------------------------------------------
# Colour palette (matches the dark CustomTkinter theme)
# ---------------------------------------------------------------------------
BG      = "#1e1e2e"
BG_MATH = "#1e1e2e"
FG      = "#cdd6f4"
FG_DIM  = "#89b4fa"
FG_REF  = "#a6e3a1"
SEP_CLR = "#45475a"


# ---------------------------------------------------------------------------
# Math rendering
# ---------------------------------------------------------------------------
_math_cache: dict[str, tk.PhotoImage] = {}


def render_math_image(
    formula: str,
    fontsize: int = 13,
    bg_color: str = BG_MATH,
    fg_color: str = "#cdd6f4",
    dpi: int = 110,
) -> tk.PhotoImage:
    """Render a mathtext formula to a tk.PhotoImage (cached).

    Uses an explicit FigureCanvasAgg so the figure doesn't need a live
    display backend — safe to call from inside a TkAgg application.
    """
    cache_key = f"{formula}|{fontsize}|{bg_color}|{dpi}"
    if cache_key in _math_cache:
        return _math_cache[cache_key]

    fig = mfig.Figure(figsize=(5, 0.7), dpi=dpi)
    FigureCanvasAgg(fig)               # attach Agg canvas explicitly
    fig.patch.set_facecolor(bg_color)

    ax = fig.add_axes([0.02, 0.05, 0.96, 0.90])
    ax.set_axis_off()
    ax.set_facecolor(bg_color)
    ax.text(
        0.5, 0.5, formula,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=fg_color,
        ha="center", va="center",
        math_fontfamily="cm",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                facecolor=bg_color, bbox_inches="tight", pad_inches=0.08)
    buf.seek(0)

    img = tk.PhotoImage(data=base64.b64encode(buf.read()).decode())
    _math_cache[cache_key] = img
    return img


# ---------------------------------------------------------------------------
# RichTooltip
# ---------------------------------------------------------------------------
class RichTooltip:
    """Attach a rich tooltip to *widget*.

    Parameters
    ----------
    widget  : any tkinter / CTk widget
    blocks  : list of block-dicts (see module docstring)
    delay_ms: hover delay before the popup appears
    """

    def __init__(self, widget, blocks: list[dict], delay_ms: int = 500):
        self._widget   = widget
        self._blocks   = blocks
        self._delay_ms = delay_ms
        self._job: str | None = None
        self._tip: tk.Toplevel | None = None

        self._bind_widget(widget)

    # ------------------------------------------------------------------
    def _bind_widget(self, widget):
        """Bind hover events, handling widgets that don't support bind() directly."""
        try:
            widget.bind("<Enter>",    self._on_enter, add="+")
            widget.bind("<Leave>",    self._on_leave, add="+")
            widget.bind("<Button-1>", self._on_leave, add="+")
        except NotImplementedError:
            # CTkSegmentedButton and similar widgets don't support bind().
            # Bind on each internal child button instead.
            children = getattr(widget, "_buttons_dict", {})
            for child in children.values():
                try:
                    child.bind("<Enter>",    self._on_enter, add="+")
                    child.bind("<Leave>",    self._on_leave, add="+")
                    child.bind("<Button-1>", self._on_leave, add="+")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _on_enter(self, event=None):
        self._cancel_job()
        self._job = self._widget.after(self._delay_ms, self._show)

    def _on_leave(self, event=None):
        self._cancel_job()
        self._hide()

    def _cancel_job(self):
        if self._job:
            self._widget.after_cancel(self._job)
            self._job = None

    # ------------------------------------------------------------------
    def _show(self):
        if self._tip and self._tip.winfo_exists():
            return

        # Position: just to the right of the cursor
        x = self._widget.winfo_rootx() + self._widget.winfo_width() + 8
        y = self._widget.winfo_rooty()

        tip = tk.Toplevel(self._widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.configure(bg=BG)
        self._tip = tip

        outer = tk.Frame(tip, bg=SEP_CLR, bd=1)
        outer.pack(fill="both", expand=True)

        inner = tk.Frame(outer, bg=BG, padx=12, pady=10)
        inner.pack(fill="both", expand=True)

        # Keep PhotoImage refs alive so GC doesn't collect them
        tip._photo_refs: list[tk.PhotoImage] = []

        for block in self._blocks:
            btype = block.get("type", "text")

            if btype == "title":
                tk.Label(
                    inner, text=block["content"],
                    bg=BG, fg="#89dceb",
                    font=("Segoe UI", 11, "bold"),
                    wraplength=380, justify="left",
                ).pack(anchor="w", pady=(0, 4))

            elif btype == "text":
                tk.Label(
                    inner, text=block["content"],
                    bg=BG, fg=FG,
                    font=("Segoe UI", 9),
                    wraplength=380, justify="left",
                ).pack(anchor="w", pady=(0, 3))

            elif btype == "math":
                try:
                    img = render_math_image(block["formula"])
                    lbl = tk.Label(inner, image=img, bg=BG, bd=0)
                    lbl.pack(anchor="center", pady=(4, 4))
                    tip._photo_refs.append(img)
                except Exception:
                    # Fallback: show raw formula as text
                    tk.Label(
                        inner, text=block["formula"],
                        bg=BG, fg=FG_DIM,
                        font=("Courier New", 9),
                    ).pack(anchor="w")

            elif btype == "ref":
                tk.Label(
                    inner, text="📖  " + block["content"],
                    bg=BG, fg=FG_REF,
                    font=("Segoe UI", 8, "italic"),
                    wraplength=380, justify="left",
                ).pack(anchor="w", pady=(2, 0))

            elif btype == "sep":
                sep = tk.Frame(inner, bg=SEP_CLR, height=1)
                sep.pack(fill="x", pady=6)

        # Clamp to screen
        tip.update_idletasks()
        sw = tip.winfo_screenwidth()
        tw = tip.winfo_width()
        if x + tw > sw - 10:
            x = self._widget.winfo_rootx() - tw - 8
        tip.wm_geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    def _hide(self):
        if self._tip and self._tip.winfo_exists():
            self._tip.destroy()
        self._tip = None


# ===========================================================================
# TOOLTIP CONTENT
# ===========================================================================
TOOLTIP_CONTENT: dict[str, list[dict]] = {

    # -----------------------------------------------------------------------
    # Posição / tempo
    # -----------------------------------------------------------------------
    "start_pos": [
        {"type": "title",   "content": "Posição Inicial do Efetuador"},
        {"type": "text",    "content": "Coordenadas cartesianas (x, y, z) em metros do ponto de partida "
                                       "da trajetória no espaço operacional."},
        {"type": "math",    "formula": r"$p_0 = [x_0,\ y_0,\ z_0]^T \in \mathbb{R}^3$"},
        {"type": "text",    "content": "A cinemática inversa é resolvida numericamente para obter "
                                       "a configuração inicial de juntas q₀."},
    ],

    "end_pos": [
        {"type": "title",   "content": "Posição Final do Efetuador"},
        {"type": "text",    "content": "Ponto de destino da trajetória no espaço cartesiano (m)."},
        {"type": "math",    "formula": r"$p_f = [x_f,\ y_f,\ z_f]^T \in \mathbb{R}^3$"},
    ],

    "init_at_start": [
        {"type": "title",   "content": "Iniciar na Posição Inicial"},
        {"type": "text",    "content": "Se marcado, as juntas partem de q₀ (IK da posição inicial) "
                                       "no instante t = 0.  Caso contrário, partem de q_init ou do último "
                                       "q convergente."},
    ],

    "total_time": [
        {"type": "title",   "content": "Tempo Total de Simulação"},
        {"type": "text",    "content": "Duração T (s) da simulação.  A trajetória é parametrizada em "
                                       "[0, T] com perfil de velocidade trapezoidal ou polinomial."},
        {"type": "math",    "formula": r"$t \in [0,\ T]$"},
    ],

    # -----------------------------------------------------------------------
    # Avançado – física
    # -----------------------------------------------------------------------
    "dt_physics": [
        {"type": "title",   "content": "Passo de Física dt"},
        {"type": "text",    "content": "Passo de integração numérica do integrador de Euler (s).  "
                                       "Valores menores aumentam a precisão mas aumentam o tempo de cômputo."},
        {"type": "math",    "formula": r"$\dot{x}_{k+1} \approx \dot{x}_k + \ddot{x}_k \cdot dt$"},
        {"type": "text",    "content": "Regra prática: dt < 1 / (10 · ω_max), onde ω_max é a "
                                       "maior frequência natural do sistema."},
        {"type": "sep"},
        {"type": "text",    "content": "Valor típico: 0.001 s (1 ms)."},
    ],

    "dt_visual": [
        {"type": "title",   "content": "Passo Visual dt"},
        {"type": "text",    "content": "Intervalo de tempo entre quadros salvos para animação (s).  "
                                       "Independente do dt de física; reduz uso de memória."},
        {"type": "text",    "content": "Valor típico: 0.05 s (20 fps).  Deve ser múltiplo inteiro "
                                       "do dt de física."},
    ],

    "q_init": [
        {"type": "title",   "content": "Configuração Inicial de Juntas q_init"},
        {"type": "text",    "content": "Vetor de ângulos (rad) separados por vírgula.  "
                                       "Deixe vazio para usar IK automática."},
        {"type": "math",    "formula": r"$q_0 = [q_1,\ q_2,\ \ldots,\ q_n]^T$"},
    ],

    "use_last_q": [
        {"type": "title",   "content": "Usar Último q Convergente"},
        {"type": "text",    "content": "Reutiliza a solução de IK da última simulação como chute "
                                       "inicial para a próxima.  Acelera a convergência quando os "
                                       "parâmetros mudam pouco entre execuções."},
    ],

    "dq_limit": [
        {"type": "title",   "content": "Limite Suave de Velocidade de Juntas"},
        {"type": "text",    "content": "Satura suavemente a velocidade de cada junta (rad/s) "
                                       "usando uma função tanh."},
        {"type": "math",    "formula": r"$\dot{q}_{sat} = \dot{q}_{lim} \cdot \tanh\!\left(\dfrac{\dot{q}}{\dot{q}_{lim}}\right)$"},
        {"type": "text",    "content": "Evita instabilidade numérica sem truncar abruptamente."},
    ],

    "feedforward_vel": [
        {"type": "title",   "content": "Feedforward de Velocidade"},
        {"type": "text",    "content": "Adiciona o termo de velocidade desejada diretamente ao "
                                       "sinal de controle, reduzindo o erro de rastreamento "
                                       "em fase (lag)."},
        {"type": "math",    "formula": r"$u = u_{fb} + \dot{q}_d$"},
    ],

    # -----------------------------------------------------------------------
    # Controle – dropdown
    # -----------------------------------------------------------------------
    "ctrl_mode": [
        {"type": "title",   "content": "Modo de Controle"},
        {"type": "text",    "content": "Seleciona a lei de controle aplicada ao robô durante a simulação."},
        {"type": "sep"},
        {"type": "text",    "content": "Torque Computado (CTC)  — linearização exata por feedback; "
                                       "requer modelo dinâmico preciso."},
        {"type": "text",    "content": "ADRC  — rejeição ativa de distúrbios; robusto a incertezas "
                                       "do modelo."},
        {"type": "text",    "content": "Sliding Mode (SMC)  — controle por modo deslizante; "
                                       "robusto e de estrutura variável."},
    ],

    # -----------------------------------------------------------------------
    # CTC
    # -----------------------------------------------------------------------
    "kp": [
        {"type": "title",   "content": "Ganho Proporcional Kp (CTC)"},
        {"type": "text",    "content": "O controle por torque computado cancela a não-linearidade "
                                       "do modelo e impõe uma dinâmica linear desejada de 2ª ordem:"},
        {"type": "math",    "formula": r"$\tau = M(q)\!\left(\ddot{q}_d + K_d\dot{e} + K_p e\right) + C(q,\dot{q})\dot{q} + g(q)$"},
        {"type": "sep"},
        {"type": "text",    "content": "Kp é equivalente ao quadrado da frequência natural:"},
        {"type": "math",    "formula": r"$K_p = \omega_n^2$"},
        {"type": "text",    "content": "Valores maiores aumentam a rigidez mas podem excitar "
                                       "vibrações ou saturar atuadores."},
        {"type": "ref",     "content": "Siciliano et al., Robotics: Modelling, Planning and Control, Springer (2009)"},
    ],

    "zeta": [
        {"type": "title",   "content": "Fator de Amortecimento ζ (CTC)"},
        {"type": "text",    "content": "Controla o regime da resposta transitória do sistema "
                                       "linearizado de 2ª ordem."},
        {"type": "math",    "formula": r"$K_d = 2\zeta\omega_n,\quad K_p = \omega_n^2$"},
        {"type": "sep"},
        {"type": "text",    "content": "ζ < 1  →  subamortecido (oscilações)"},
        {"type": "text",    "content": "ζ = 1  →  criticamente amortecido (sem sobressinal)"},
        {"type": "text",    "content": "ζ > 1  →  superamortecido (resposta lenta)"},
        {"type": "text",    "content": "Recomendado: ζ = 1.0 para rastreamento sem sobressinal."},
        {"type": "ref",     "content": "Siciliano et al., Robotics: Modelling, Planning and Control, Springer (2009)"},
    ],

    # -----------------------------------------------------------------------
    # ADRC
    # -----------------------------------------------------------------------
    "omega_c": [
        {"type": "title",   "content": "ωc — Largura de Banda do Controlador (ADRC)"},
        {"type": "text",    "content": "O ADRC (Active Disturbance Rejection Control) usa um "
                                       "Observador de Estado Estendido (ESO) para estimar e cancelar "
                                       "perturbações em tempo real."},
        {"type": "math",    "formula": r"$u = \frac{1}{b_0}\!\left[u_0 - \hat{z}_3\right],\quad "
                                        r"u_0 = K_p(r - \hat{z}_1) - K_d\hat{z}_2$"},
        {"type": "sep"},
        {"type": "text",    "content": "ωc define os pólos do controlador (em rad/s):"},
        {"type": "math",    "formula": r"$K_p = \omega_c^2,\quad K_d = 2\omega_c$"},
        {"type": "text",    "content": "Aumentar ωc torna o sistema mais rápido mas mais "
                                       "sensível a ruído e atraso de medição."},
        {"type": "ref",     "content": "Han, J. — From PID to Active Disturbance Rejection Control, IEEE Trans. Ind. Electron. (2009)"},
    ],

    "omega_o": [
        {"type": "title",   "content": "ωo — Largura de Banda do Observador ESO (ADRC)"},
        {"type": "text",    "content": "Define a velocidade com que o ESO (Extended State Observer) "
                                       "converge para os estados reais e para o distúrbio total."},
        {"type": "math",    "formula": r"$\dot{\hat{z}} = A\hat{z} + Bu + L(y - \hat{z}_1)$"},
        {"type": "sep"},
        {"type": "text",    "content": "Regra de sintonização:"},
        {"type": "math",    "formula": r"$\omega_o \geq 5\,\omega_c$"},
        {"type": "text",    "content": "Observadores muito rápidos amplificam ruído.  "
                                       "Típico: ωo = 3–10 × ωc."},
        {"type": "ref",     "content": "Gao, Z. — Scaling and Bandwidth-Parameterization Based Controller Tuning, ACC (2003)"},
    ],

    "gravity_ff": [
        {"type": "title",   "content": "Feedforward de Gravidade G(q)"},
        {"type": "text",    "content": "Adiciona o torque de compensação de gravidade calculado "
                                       "pelo modelo antes de enviar ao ESO, reduzindo a carga "
                                       "de estimação do observador."},
        {"type": "math",    "formula": r"$\tau = \tau_{ADRC} + g(q)$"},
        {"type": "text",    "content": "Recomendado quando o modelo de gravidade é preciso."},
    ],

    "coriolis_ff": [
        {"type": "title",   "content": "Feedforward de Coriolis C(q, dq)"},
        {"type": "text",    "content": "Injeta o termo de Coriolis/centrífugo do modelo dinâmico "
                                       "como feedforward, aliviando o observador."},
        {"type": "math",    "formula": r"$\tau = \tau_{ADRC} + C(q,\dot{q})\dot{q}$"},
        {"type": "text",    "content": "Mais relevante em velocidades altas."},
    ],

    "auto_b0": [
        {"type": "title",   "content": "Auto b0"},
        {"type": "text",    "content": "b0 é o ganho nominal de entrada do modelo (inverso da inércia "
                                       "do atuador).  Com Auto b0 ativo, é calculado automaticamente "
                                       "a partir da diagonal da matriz de inércia M(q) na configuração inicial:"},
        {"type": "math",    "formula": r"$b_0 \approx \dfrac{1}{M_{ii}(q_0)}$"},
        {"type": "text",    "content": "Desmarque para ajustar manualmente (útil quando o modelo "
                                       "subestima a inércia real)."},
    ],

    "b0": [
        {"type": "title",   "content": "b0 — Ganho de Entrada (ADRC)"},
        {"type": "text",    "content": "Parâmetro de escala que relaciona a entrada de controle u "
                                       "com a aceleração do sistema:"},
        {"type": "math",    "formula": r"$\ddot{q} \approx b_0 u + f_{total}$"},
        {"type": "text",    "content": "Um b0 errado não impede a estabilidade (o ESO compensa), "
                                       "mas degrada o desempenho transitório."},
    ],

    # ADRC avançado
    "z_limit": [
        {"type": "title",   "content": "Limite de Saturação dos Estados z do ESO"},
        {"type": "text",    "content": "Satura os estados internos do observador para evitar "
                                       "wind-up em presença de não-linearidades severas ou "
                                       "condições iniciais muito erradas."},
        {"type": "math",    "formula": r"$\hat{z}_i \leftarrow \mathrm{clip}(\hat{z}_i,\ -z_{lim},\ z_{lim})$"},
    ],

    "tau_limit": [
        {"type": "title",   "content": "Limite de Torque τ (Nm)"},
        {"type": "text",    "content": "Satura o torque de saída do controlador ADRC para "
                                       "respeitar limites físicos do atuador."},
        {"type": "math",    "formula": r"$\tau_{cmd} = \mathrm{clip}(\tau,\ -\tau_{lim},\ \tau_{lim})$"},
    ],

    "max_wo_dt": [
        {"type": "title",   "content": "Máximo ωo · dt (Estabilidade do ESO)"},
        {"type": "text",    "content": "Limita o produto ωo × dt para garantir estabilidade "
                                       "numérica do ESO discretizado por Euler explícito."},
        {"type": "math",    "formula": r"$\omega_o \cdot dt \leq 0.1\ \text{(regra prática)}$"},
        {"type": "text",    "content": "Se ωo × dt exceder esse limite o código reduz ωo "
                                       "automaticamente naquele passo."},
    ],

    "tau_filter_alpha": [
        {"type": "title",   "content": "Filtro de Torque α (EMA)"},
        {"type": "text",    "content": "Aplica um filtro de média móvel exponencial ao torque "
                                       "de saída para reduzir chattering de alta frequência:"},
        {"type": "math",    "formula": r"$\tau_k = \alpha\,\tau_{k-1} + (1-\alpha)\,\tau_{raw}$"},
        {"type": "text",    "content": "α ≈ 0.8  →  filtragem moderada.  α = 0  →  sem filtro."},
    ],

    "z3_filter_alpha": [
        {"type": "title",   "content": "Filtro do Estado z₃ α (EMA)"},
        {"type": "text",    "content": "Filtra z₃ (estimativa do distúrbio total) antes de usar "
                                       "na lei de controle.  Reduz ruído amplificado pelo observador "
                                       "rápido."},
        {"type": "math",    "formula": r"$\hat{z}_{3,k}^{filt} = \alpha\,\hat{z}_{3,k-1}^{filt} + (1-\alpha)\,\hat{z}_{3,k}$"},
        {"type": "text",    "content": "α ≈ 0.2  →  resposta rápida com leve suavização."},
    ],

    # -----------------------------------------------------------------------
    # SMC
    # -----------------------------------------------------------------------
    "smc_lambda": [
        {"type": "title",   "content": "Lambda λ — Inclinação da Superfície Deslizante (SMC)"},
        {"type": "text",    "content": "O controle por modo deslizante força o estado do sistema "
                                       "a atingir e permanecer em uma superfície s = 0 no espaço "
                                       "de estados."},
        {"type": "math",    "formula": r"$s = \dot{e} + \lambda\, e$"},
        {"type": "sep"},
        {"type": "text",    "content": "λ determina a dinâmica sobre a superfície (pólo em −λ).  "
                                       "Valores maiores → convergência mais rápida do erro, "
                                       "mas mais sensível a ruído."},
        {"type": "ref",     "content": "Slotine & Li, Applied Nonlinear Control, Prentice-Hall (1991)"},
    ],

    "smc_k": [
        {"type": "title",   "content": "Ganho K — Chaveamento (SMC)"},
        {"type": "text",    "content": "Lei de controle por chaveamento que garante a condição "
                                       "de alcançabilidade:"},
        {"type": "math",    "formula": r"$\dot{V} = s\dot{s} \leq -\eta\,|s|,\quad \eta > 0$"},
        {"type": "sep"},
        {"type": "text",    "content": "Lei de controle equivalente + chaveamento:"},
        {"type": "math",    "formula": r"$u = u_{eq} - K\,\mathrm{sat}(s/\phi)$"},
        {"type": "text",    "content": "K deve ser maior que a perturbação máxima esperada.  "
                                       "Valores excessivos causam chattering."},
        {"type": "ref",     "content": "Slotine & Li, Applied Nonlinear Control, Prentice-Hall (1991)"},
    ],

    "smc_phi": [
        {"type": "title",   "content": "Camada Limite ϕ (SMC)"},
        {"type": "text",    "content": "Substitui o sinal sgn(s) pela função saturação "
                                       "dentro de uma faixa ±ϕ em torno da superfície, "
                                       "eliminando o chattering:"},
        {"type": "math",    "formula": r"$\mathrm{sat}(s/\phi) = s/\phi \quad \text{se } |s| \leq \phi$"},
        {"type": "math",    "formula": r"$\mathrm{sat}(s/\phi) = \mathrm{sgn}(s) \quad \text{se } |s| > \phi$"},
        {"type": "text",    "content": "ϕ pequeno → menor erro em regime, mais chattering.  "
                                       "ϕ grande → sem chattering, erro em regime maior."},
        {"type": "ref",     "content": "Slotine & Li, Applied Nonlinear Control, Prentice-Hall (1991)"},
    ],

    # -----------------------------------------------------------------------
    # Perturbação
    # -----------------------------------------------------------------------
    "disturbance": [
        {"type": "title",   "content": "Perturbação Constante nas Juntas (Nm)"},
        {"type": "text",    "content": "Adiciona um torque externo constante em todas as juntas "
                                       "durante toda a simulação.  Útil para testar a robustez "
                                       "do controlador."},
        {"type": "math",    "formula": r"$\tau_{total} = \tau_{ctrl} + \tau_{dist}$"},
        {"type": "text",    "content": "Faixa: −20 a +20 Nm.  Zero = sem perturbação."},
    ],

    # -----------------------------------------------------------------------
    # Trajetória
    # -----------------------------------------------------------------------
    "traj_type": [
        {"type": "title",   "content": "Tipo de Trajetória"},
        {"type": "text",    "content": "Reta  — interpolação linear entre p₀ e pf no espaço "
                                       "cartesiano com perfil de velocidade suave."},
        {"type": "text",    "content": "Círculo  — trajetória circular de raio R em torno do "
                                       "ponto médio, no plano definido pelo vetor normal."},
    ],

    "radius": [
        {"type": "title",   "content": "Raio da Trajetória Circular (m)"},
        {"type": "math",    "formula": r"$p(t) = p_c + R\!\left[\cos\!\left(\theta(t)\right)\hat{u} + "
                                        r"\sin\!\left(\theta(t)\right)\hat{v}\right]$"},
        {"type": "text",    "content": "onde p_c é o centro, R o raio e {û, v̂} são vetores "
                                       "ortogonais no plano do círculo."},
    ],

    "normal": [
        {"type": "title",   "content": "Vetor Normal ao Plano do Círculo"},
        {"type": "text",    "content": "Define a orientação do plano em que o círculo está contido.  "
                                       "O vetor é normalizado automaticamente."},
        {"type": "math",    "formula": r"$\hat{n} = \frac{n}{\|n\|}$"},
        {"type": "text",    "content": "Exemplos:  (1,0,0) → plano YZ,  (0,0,1) → plano XY."},
    ],

    "direction": [
        {"type": "title",   "content": "Sentido de Percurso"},
        {"type": "text",    "content": "Anti-Horário (+1): θ cresce de 0 a 2π (regra da mão direita "
                                       "relativa ao vetor normal)."},
        {"type": "text",    "content": "Horário (−1): θ decresce (sentido inverso)."},
        {"type": "math",    "formula": r"$\theta(t) = \pm\,\frac{2\pi}{T}\,t$"},
    ],

    # -----------------------------------------------------------------------
    # Botões de ação – simulação
    # -----------------------------------------------------------------------
    "adv_basic_toggle": [
        {"type": "title",   "content": "Seção Avançada — Física"},
        {"type": "text",    "content": "Expande ou recolhe os parâmetros técnicos do integrador: "
                                       "passo de física, passo visual, configuração inicial de juntas "
                                       "e limites de velocidade."},
        {"type": "text",    "content": "Os valores padrão são adequados para a maioria dos casos. "
                                       "Edite apenas se precisar de maior precisão ou estabilidade numérica."},
    ],

    "adrc_adv_toggle": [
        {"type": "title",   "content": "Seção Avançada — ADRC"},
        {"type": "text",    "content": "Expande parâmetros de proteção do controlador ADRC: "
                                       "saturações dos estados do observador, limite de torque, "
                                       "estabilidade numérica e filtros de suavização."},
        {"type": "text",    "content": "Útil para robôs com dinâmica agressiva ou em presença "
                                       "de ruído elevado nos sensores."},
    ],

    "btn_restore_defaults": [
        {"type": "title",   "content": "Restaurar Parâmetros Físicos Padrão"},
        {"type": "text",    "content": "Redefine todos os campos de Massa, Comprimento, Centro de Massa "
                                       "e Tensor de Inércia de cada elo para os valores padrão "
                                       "carregados no modelo atual."},
        {"type": "text",    "content": "Útil para desfazer edições manuais sem precisar reiniciar."},
    ],

    "btn_run_sim": [
        {"type": "title",   "content": "Rodar Simulação"},
        {"type": "text",    "content": "Executa a simulação dinâmica com os parâmetros configurados. "
                                       "O processo roda em background para não travar a interface."},
        {"type": "text",    "content": "Ao final, os gráficos de posição, velocidade, torque e erro "
                                       "são exibidos no painel direito."},
        {"type": "math",    "formula": r"$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau_{ctrl} + \tau_{dist}$"},
    ],

    "btn_anim3d": [
        {"type": "title",   "content": "Ver Animação 3D"},
        {"type": "text",    "content": "Abre uma janela com a animação tridimensional do movimento "
                                       "do robô simulado, usando os dados calculados."},
        {"type": "text",    "content": "Disponível somente após rodar a simulação com sucesso."},
    ],

    # -----------------------------------------------------------------------
    # Botões de ação – modelagem
    # -----------------------------------------------------------------------
    "mode_switch": [
        {"type": "title",   "content": "Ambiente de Operação"},
        {"type": "text",    "content": "Ar (Seco): gera o modelo dinâmico padrão com inércia, "
                                       "Coriolis e gravidade."},
        {"type": "math",    "formula": r"$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau$"},
        {"type": "sep"},
        {"type": "text",    "content": "Água (UVMS): adiciona efeitos hidrodinâmicos — empuxo, "
                                       "arrasto e massa adicionada — para robôs subaquáticos."},
        {"type": "math",    "formula": r"$[M(q)+M_A]\ddot{q} + \cdots + D(q,\dot{q}) + g_{buoy}(q) = \tau$"},
        {"type": "ref",     "content": "Fossen, T.I. — Handbook of Marine Craft Hydrodynamics and Motion Control (2011)"},
    ],

    "btn_add_joint": [
        {"type": "title",   "content": "Adicionar Junta"},
        {"type": "text",    "content": "Insere uma nova junta na cadeia cinemática.  "
                                       "Configure o tipo de junta e o eixo do elo associado."},
        {"type": "sep"},
        {"type": "text",    "content": "Tipos de junta disponíveis:"},
        {"type": "text",    "content": "Rz / Ry / Rx  →  revolução em torno do eixo Z / Y / X"},
        {"type": "text",    "content": "Dz / Dy / Dx  →  translação (prismática) ao longo de Z / Y / X"},
        {"type": "text",    "content": "Elo (L): marque os eixos que têm comprimento não nulo."},
    ],

    "btn_rem_joint": [
        {"type": "title",   "content": "Remover Última Junta"},
        {"type": "text",    "content": "Remove a junta mais recente da cadeia cinemática.  "
                                       "O modelo mínimo é de 1 junta."},
    ],

    # -----------------------------------------------------------------------
    # Parâmetros físicos – entradas dinâmicas por elo
    # -----------------------------------------------------------------------
    "phys_mass": [
        {"type": "title",   "content": "Massa do Elo (kg)"},
        {"type": "text",    "content": "Massa total do elo, concentrada no centro de massa.  "
                                       "Afeta a matriz de inércia M(q) e o vetor de gravidade g(q)."},
        {"type": "math",    "formula": r"$M(q) = \sum_{i} \left[m_i J_{v_i}^T J_{v_i} + J_{\omega_i}^T I_i J_{\omega_i}\right]$"},
        {"type": "text",    "content": "Valores típicos para robôs industriais: 1–10 kg por elo."},
    ],

    "phys_length": [
        {"type": "title",   "content": "Comprimento do Elo (m)"},
        {"type": "text",    "content": "Comprimento geométrico do elo ao longo do eixo ativo.  "
                                       "Determina o deslocamento entre juntas consecutivas na "
                                       "cinemática direta."},
        {"type": "math",    "formula": r"$p_{i+1} = p_i + R_i\, L_i\,\hat{e}$"},
        {"type": "text",    "content": "Apenas o componente do eixo marcado em 'Elo (L)' é editável; "
                                       "os demais são fixos em 0."},
    ],

    "phys_com": [
        {"type": "title",   "content": "Centro de Massa do Elo (m)"},
        {"type": "text",    "content": "Posição do centro de massa relativa à origem da junta "
                                       "do elo, expressa no referencial local."},
        {"type": "math",    "formula": r"$r_{c_i} = [c_x,\ c_y,\ c_z]^T$"},
        {"type": "text",    "content": "Afeta o vetor de gravidade g(q) e o termo de Coriolis C(q,q̇).  "
                                       "Para barras uniformes: c = L/2."},
    ],

    "phys_inertia": [
        {"type": "title",   "content": "Tensor de Inércia — Diagonal (kg·m²)"},
        {"type": "text",    "content": "Momentos de inércia principais em relação ao centro de massa, "
                                       "no referencial do elo.  A matriz completa é simétrica; "
                                       "os termos fora da diagonal são assumidos nulos."},
        {"type": "math",    "formula": r"$I_i = \mathrm{diag}(I_{xx},\ I_{yy},\ I_{zz})$"},
        {"type": "text",    "content": "Para uma barra cilíndrica uniforme (massa m, raio r, comprimento L):"},
        {"type": "math",    "formula": r"$I_{xx} = I_{yy} = \tfrac{1}{12}m(3r^2+L^2)$"},
        {"type": "math",    "formula": r"$I_{zz} = \tfrac{1}{2}m r^2$"},
        {"type": "ref",     "content": "Siciliano et al., Robotics: Modelling, Planning and Control, Springer (2009) — Apêndice B"},
    ],

    "phys_volume": [
        {"type": "title",   "content": "Volume do Elo (m³)  — Modo Hidro"},
        {"type": "text",    "content": "Volume deslocado pelo elo, usado para calcular o empuxo "
                                       "hidrostático de Arquimedes:"},
        {"type": "math",    "formula": r"$F_{buoy} = \rho g V$"},
        {"type": "text",    "content": "Para um cilindro: V = π r² L.  "
                                       "Valores típicos: 0.001–0.01 m³."},
    ],

    "phys_added_mass_lin": [
        {"type": "title",   "content": "Massa Adicionada Linear (kg)  — Modo Hidro"},
        {"type": "text",    "content": "Coeficientes de massa adicionada nas direções lineares "
                                       "(u, v, w).  Modelam a inércia do fluido acelerado ao redor do elo."},
        {"type": "math",    "formula": r"$M_{eff} = M_{elo} + M_A,\quad M_A = \mathrm{diag}(m_{a_u}, m_{a_v}, m_{a_w})$"},
        {"type": "text",    "content": "Para um cilindro com eixo ao longo de u:  "
                                       "m_a_u ≈ 0,  m_a_v = m_a_w ≈ ρ π r² L."},
        {"type": "ref",     "content": "Fossen, T.I. — Handbook of Marine Craft Hydrodynamics (2011)"},
    ],

    "phys_added_mass_ang": [
        {"type": "title",   "content": "Massa Adicionada Angular (kg·m²)  — Modo Hidro"},
        {"type": "text",    "content": "Coeficientes de inércia adicionada nas direções rotacionais "
                                       "(p, q, r).  Geralmente pequenos para elos subaquáticos típicos."},
        {"type": "math",    "formula": r"$I_{eff} = I_{elo} + I_A,\quad I_A = \mathrm{diag}(I_{a_p}, I_{a_q}, I_{a_r})$"},
    ],

    "phys_rho": [
        {"type": "title",   "content": "Densidade do Fluido ρ (kg/m³)"},
        {"type": "text",    "content": "Densidade do fluido em que o robô opera.  "
                                       "Usada no cálculo do empuxo e do arrasto viscoso."},
        {"type": "math",    "formula": r"$F_{buoy} = \rho\, g\, V$"},
        {"type": "text",    "content": "Água doce: 1000 kg/m³  |  Água do mar: ≈ 1025 kg/m³"},
    ],

    "btn_calc": [
        {"type": "title",   "content": "Gerar Modelo Dinâmico"},
        {"type": "text",    "content": "Calcula simbolicamente, via Lagrangeano, as matrizes "
                                       "M(q), C(q,q̇) e g(q) para a cadeia cinemática configurada."},
        {"type": "math",    "formula": r"$\mathcal{L} = T(q,\dot{q}) - V(q)$"},
        {"type": "math",    "formula": r"$\tau_i = \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i}$"},
        {"type": "text",    "content": "As equações são então compiladas numericamente (NumPy/lambdify) "
                                       "para uso no simulador."},
        {"type": "ref",     "content": "Siciliano et al., Robotics: Modelling, Planning and Control, Springer (2009)"},
    ],
}


# ---------------------------------------------------------------------------
# Helper: resolve the right tooltip blocks for a dynamic physical-entry key
# ---------------------------------------------------------------------------
import re as _re

_PHYS_KEY_MAP = [
    (_re.compile(r"^m\d+$"),         "phys_mass"),
    (_re.compile(r"^L\d+$"),         "phys_length"),
    (_re.compile(r"^c[xyz]\d+$"),    "phys_com"),
    (_re.compile(r"^I[xyz]{2}\d+$"), "phys_inertia"),
    (_re.compile(r"^vol\d+$"),       "phys_volume"),
    (_re.compile(r"^ma_[uvw]\d+$"),  "phys_added_mass_lin"),
    (_re.compile(r"^ma_[pqr]\d+$"),  "phys_added_mass_ang"),
    (_re.compile(r"^rho$"),          "phys_rho"),
]


def phys_tooltip_blocks(key: str) -> list[dict] | None:
    """Return tooltip blocks for a dynamic physical-parameter entry key, or None."""
    for pattern, content_key in _PHYS_KEY_MAP:
        if pattern.match(key):
            return TOOLTIP_CONTENT[content_key]
    return None
