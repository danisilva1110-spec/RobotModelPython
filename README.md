# SUMÉ

**Simulador Unificado de Manipuladores e Estratégias de Controle** — plataforma Python open-source para modelagem simbólica, simulação e controle de manipuladores robóticos e UVMSs (Underwater Vehicle-Manipulator Systems).

Desenvolvido como Projeto Final do Curso de Engenharia de Controle e Automação — CEFET-RJ.

> *Na mitologia tupi, Sumé é o herói civilizador que ensinou técnicas e conhecimento ao povo. O SUMÉ forja modelos dinâmicos exatos para manipuladores robóticos em um simulador unificado com múltiplas estratégias de controle.*

---

## Visão Geral

O SUMÉ resolve o gargalo da transição simbólico-numérica em plataformas abertas para robótica subaquática. Dado uma cadeia cinemática arbitrária, ele deriva automaticamente as matrizes dinâmicas completas e executa simulação de alta fidelidade com quatro estratégias de controle não-linear.

### Funcionalidades Principais

- **Modelagem Simbólica (Euler-Lagrange):** derivação automática de M(q), C(q,q̇), G(q) e J(q) para N graus de liberdade seriais
- **Extensão Hidrodinâmica:** massa adicionada por elo e potencial aparente Peso-Empuxo para ambiente subaquático (UVMS)
- **Otimização Computacional:** `lambdify + CSE` (60–90% redução de bytecode) e paralelismo em dois níveis via `ProcessPoolExecutor`
- **Simulação Numérica:** integrador de Euler, trajetórias cúbicas, forças dissipativas (atrito Coulomb+viscoso, arrasto hidrodinâmico)
- **CLIK 6D Completo:** Jacobiano 6×N, SLERP geodésico em SO(3), DLS, espaço nulo, Levenberg-Marquardt
- **Quatro Controladores Não-Lineares:** CTC-PID, CT-SMC, Super-Twisting (STA) e LADRC em interface unificada
- **Interface Gráfica:** configuração interativa via `customtkinter`, análise comparativa de múltiplas sessões

---

## Instalação

**Requisitos:** Python 3.10+

```bash
git clone https://github.com/danisilva1110-spec/RobotModelPython.git
cd RobotModelPython
pip install -r requirements.txt
python main.py
```

---

## Uso Rápido

1. **Aba Modelagem:** configure a cadeia cinemática (tipos de junta e direções dos elos) e clique em *Gerar Modelo*
2. **Aba Simulação:** defina parâmetros físicos, trajetória, controlador e execute
3. **Aba Análise:** sobreponha múltiplas simulações para comparação de controladores

### Exemplo — SCARA 4 GDL

| Junta | Tipo | Elo |
|-------|------|-----|
| 1 | Revoluta Z | X: 0.5 m |
| 2 | Revoluta Z | X: 0.5 m |
| 3 | Prismática Z | Z: 0.5 m |
| 4 | Revoluta Z | — |

---

## Arquitetura

```
RobotModelPython/
├── engine.py            # Derivação simbólica (Euler-Lagrange, Christoffel, Hidro)
├── simulator.py         # Integração numérica, CLIK, controladores
├── main.py              # Interface gráfica (customtkinter)
├── tooltip_utils.py
├── benchmark_parallel.py
└── requirements.txt
```

---

## Controladores Implementados

| Controlador | Descrição | Indicado para |
|-------------|-----------|---------------|
| **CTC-PID** | Torque Computado + PID | Modelo exato, ambiente controlado |
| **CT-SMC** | Modo Deslizante com CTC | Perturbações limitadas conhecidas |
| **STA** | Super-Twisting de 2ª ordem | Atuadores DC, sinal contínuo |
| **LADRC** | Rejeição Ativa de Distúrbios | UVMSs, incerteza hidrodinâmica ≥ 30% |

---

## Resultados

Validado em UVMS de 12 GDL (veículo 6 GDL + braço antropomórfico 6R) em ambiente subaquático com massa adicionada e empuxo por elo. Trajetória: segmento linear + arco circular 3D com orientação SLERP.

---

## Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um **fork** do repositório
2. Crie uma branch: `git checkout -b feature/minha-contribuicao`
3. Faça suas alterações e commit: `git commit -m "Adiciona: descrição"`
4. Abra um **Pull Request** com descrição detalhada

### Ideias para contribuições futuras

- [ ] Interface serial UART/USB com microcontroladores
- [ ] Import de geometria CAD (STEP) com extração automática de m, CM, I
- [ ] Integração com ROS2
- [ ] Suporte a robôs paralelos (Stewart, Delta)
- [ ] Geração de datasets para aprendizado por reforço (RLlib)
- [ ] Geração de código C embarcado
- [ ] Suporte a AUV/ROV com propulsores
- [ ] Geração de `.exe` com suporte a multiprocessamento (`freeze_support`)

---

## Citação

Se você usar o SUMÉ em sua pesquisa, por favor cite:

```bibtex
@mastersthesis{santos2026sume,
  author  = {Lucas da Silva Santos},
  title   = {Simulação de Estratégias de Controle para Estabilização
             de Manipuladores Subaquáticos com Múltiplos Graus de Liberdade},
  school  = {Centro Federal de Educação Tecnológica Celso Suckow da Fonseca},
  year    = {2026},
  type    = {Projeto Final de Graduação}
}
```
