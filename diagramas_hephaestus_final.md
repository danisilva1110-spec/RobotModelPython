# Diagramas de Fluxo — TCC: Hephaestus
## Plataforma de Simulação e Controle de Manipuladores Robóticos

> **Como renderizar:** [Mermaid Live Editor](https://mermaid.live) ≥ v10.4

---

## Diagrama 1 — Arquitetura Geral do Sistema

```mermaid
flowchart TD
    USER(["Usuário / GUI"])

    subgraph ENGINE["engine.py — Geração Simbólica"]
        E1["Configuração da cadeia cinemática"]
        E2["Cinemática direta simbólica"]
        E3["Inércia, Gravidade e Jacobiano"]
        E4["Coriolis via Christoffel"]
        E5["Exporta modelo  {M, C, G, J}"]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    subgraph SIM["simulator.py — Simulação Numérica"]
        S1["Compila modelo para NumPy"]
        S2["Planeja trajetória cartesiana"]
        S3["Cinemática inversa — CLIK"]
        S4["Avalia M, C, G numericamente"]
        S5["Calcula torque de controle"]
        S6["Integra equações de movimento"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
        S6 -->|"q, dq"| S2
    end

    USER -->|"Topologia + parâmetros"| E1
    E5 -->|"Modelo simbólico"| S1
    S6 -->|"Resultados + animação"| USER
```

---

## Diagrama 2a — Cinemática Simbólica

```mermaid
flowchart TD
    A(["step_1_kinematics()"])
    B["Cria símbolos para cada junta<br/>q<sub>i</sub>, m<sub>i</sub>, L<sub>i</sub>, c<sub>xi</sub>, c<sub>yi</sub>, c<sub>zi</sub>"]

    C{Tipo de junta?}
    DR["Revolução<br/>Atualiza rotação e velocidade angular"]
    DD["Prismática<br/>Atualiza translação; rotação inalterada"]

    E["Acumula transformação homogênea $$T_i$$"]
    F["Calcula posição do CM no frame global"]

    G{Mais juntas?}
    H(["Saída: $$T_i,\; p_{cm_i},\; \omega_i$$"])

    A --> B --> C
    C -->|R| DR --> E
    C -->|D| DD --> E
    E --> F --> G
    G -->|Sim| B
    G -->|Não| H
```

---

## Diagrama 2b — Dinâmica Simbólica: M, G e Coriolis

```mermaid
flowchart TD
    A(["Entrada: $$T_i,\; p_{cm_i},\; \omega_i,\; R_i$$"])

    subgraph P2["Passo 2 — Inércia e Gravidade"]
        B["Calcula Jacobianos do CM<br/>para cada elo"]
        C["Transforma tensor de inércia<br/>para frame global"]
        D["Acumula matriz de inércia $$M(q)$$"]
        E["Acumula energia potencial $$V(q)$$"]
        F["$$\text{Deriva } G(q) \text{ e } J(q) \text{ simbolicamente}$$"]
        B --> C --> D --> E --> F
    end

    subgraph P3["Passo 3 — Coriolis"]
        G{Paralelismo<br/>disponível?}
        GP["Diferencia M em relação<br/>a cada q<sub>k</sub> em paralelo"]
        GS["Diferencia M em relação<br/>a cada q<sub>k</sub> em série"]
        H["Monta C(q, q̇) pelos<br/>símbolos de Christoffel"]
        G -->|Sim| GP --> H
        G -->|Não| GS --> H
    end

    A --> P2 --> P3
    P3 --> Z(["Saída: $$M(q),\; C(q,\dot{q}),\; G(q),\; J(q)$$"])
```

---

## Diagrama 3 — Extensão Hidrodinâmica

```mermaid
flowchart TD
    A(["RobotMathHydro"])

    B["Herda cinemática padrão<br/>Adiciona: densidade ρ e V<sub>i</sub>"]

    subgraph DIN["Dinâmica subaquática"]
        C["Cria coeficientes de massa adicionada<br/>por translação e rotação"]
        D["Transforma massa adicionada<br/>para frame global"]
        E["Acumula inércia aumentada<br/>(corpo + fluido acelerado)"]
        F["Energia potencial aparente<br/>Peso − Empuxo por elo"]
        G["Deriva G<sub>hydro</sub>(q)"]
        C --> D --> E --> F --> G
    end

    H{Elo neutralmente<br/>flutuante?}
    I["G<sub>i</sub> ≈ 0<br/>Sem torque estático"]
    J["G<sub>i</sub> ≠ 0<br/>Torque estático residual"]

    K["Coriolis herdado<br/>(mesmos Christoffel sobre M<sub>hydro</sub>)"]
    L(["Exporta com Mode = 'Hydro'"])

    A --> B --> DIN --> H
    H -->|Sim| I --> K
    H -->|Não| J --> K
    K --> L
```

---

## Diagrama 4 — Compilação Simbólico-Numérica

```mermaid
flowchart LR
    A(["Modelo simbólico<br/>{M, C, G, J, frames}"])

    subgraph CSE["Otimização — CSE"]
        C1["Identifica subexpressões repetidas"]
        C2["Substitui por variáveis temporárias"]
        C3["Código 2–5× mais rápido"]
        C1 --> C2 --> C3
    end

    subgraph FUNCS["Funções NumPy compiladas"]
        F1["func_M  →  Inércia"]
        F2["func_C  →  Coriolis"]
        F3["func_G  →  Gravidade"]
        F4["func_J  →  Jacobiano"]
        F5["funcs_fk[ ]  →  Posição dos elos"]
        F6["func_R_last  →  Orientação final"]
    end

    D{UVMS?}
    E["+ func_R_vehicle<br/>Orientação do veículo"]
    FIM(["Simulador pronto"])

    A --> CSE --> FUNCS --> D
    D -->|Sim| E --> FIM
    D -->|Não| FIM
```

---

## Diagrama 5 — Planejamento de Trajetória

```mermaid
flowchart TD
    A(["trajectory_planning(t, Pi, Pf, modo)"])

    B{Tempo<br/>esgotado?}
    C(["Mantém posição final<br/>com velocidade zero"])

    D["Perfil cúbico de velocidade<br/>s(t) = 3t<sup>2</sup> − 2t<sup>3</sup>"]

    E{Forma da<br/>trajetória?}
    EL["Linha reta<br/>Interpolação linear com perfil s(t)"]
    EC["Arco circular<br/>Parametriza ângulo com perfil s(t)"]

    F["Calcula orientação de referência<br/>conforme modo selecionado"]
    G(["Saída: $$P_{ref},\; V_{ref},\; A_{ref},\; R_{ref},\; \omega_{ref}$$"])

    A --> B
    B -->|Sim| C
    B -->|Não| D --> E
    E -->|Linha| EL --> F
    E -->|Círculo| EC --> F
    F --> G
```

---

## Diagrama 6a — CLIK: Cinemática Inversa Numérica

```mermaid
flowchart TD
    A(["Entrada: $$P_{ref},\; V_{ref},\; A_{ref},\; q,\; \dot{q}$$"])

    B["Calcula posição atual via FK<br/>e<sub>p</sub> = P<sub>ref</sub> − p<sub>atual</sub>"]

    C["Avalia J(q)<br/>Estima deriva: J̇q̇"]

    D{Com controle<br/>de orientação?}
    E6["Tarefa 6D<br/>v = [ṗ<sub>ref</sub> ; ω<sub>ref</sub>]<br/>Jacobiano pseudo-inverso 6×N"]
    E3["Tarefa 3D<br/>v = ṗ<sub>ref</sub><br/>Jacobiano pseudo-inverso 3×N"]

    F["Calcula $$\dot{q}_d,\; \ddot{q}_d$$ de referência"]
    G["Adiciona componente no espaço nulo<br/>para preferência de postura"]
    H["Satura $$\dot{q} com função \tanh$$"]

    I(["Saída: $$q_d,\; \dot{q}_d,\; \ddot{q}_d$$"])

    A --> B --> C --> D
    D -->|Sim| E6 --> F
    D -->|Não| E3 --> F
    F --> G --> H --> I
```

---

## Diagrama 6b — Modos de Orientação do Efetuador

```mermaid
flowchart TD
    A{orient_mode?}

    BL["Livre<br/>Sem restrição de orientação"]
    BF["Fixa<br/>Mantém orientação inicial"]
    BT["Tangente à Trajetória<br/>Eixo Z alinhado com a velocidade"]
    BO["Apontar para o Alvo<br/>Eixo Z aponta para P<sub>f</sub>"]
    BS["SLERP<br/>Interpolação esférica R<sub>0</sub> → R<sub>f</sub>"]
    BN["Normal à Superfície<br/>Eixo Z alinhado com a normal"]

    C["Calcula erro de orientação<br/>e<sub>R</sub> = skew(R<sub>ref</sub>, R<sub>atual</sub>)"]
    D["Gera comando angular<br/>ω<sub>cmd</sub> = ω<sub>ref</sub> + K<sub>p</sub> e<sub>R</sub>"]

    A -->|Livre| BL
    A -->|Fixa| BF
    A -->|Tangente| BT
    A -->|Alvo| BO
    A -->|SLERP| BS
    A -->|Normal| BN

    BT & BO & BS & BN --> C --> DSim
    
    Não
    
    
```

---

## Diagrama 7a — Loop de Simulação: Inicialização

```mermaid
flowchart TD
    A(["run(t_total, Pi, Pf, ctrl_params)"])

    B["Define passos de integração<br/>dt_phys, dt_visual, substeps"]
    C["IK inicial — Levenberg-Marquardt<br/>posiciona robô em P<sub>i</sub>"]

    D{Convergiu?}
    E["Inicia em P<sub>i</sub><br/>q = q<sub>IK</sub>,  q̇ = 0"]
    F["Aviso: usa postura home<br/>q = q<sub>home</sub>,  q̇ = 0"]

    G{Controlador<br/>selecionado?}
    G1["CTC-PID<br/>K<sub>P</sub>, K<sub>D</sub>, K<sub>I</sub>"]
    G2["CT-SMC<br/>λ, K, φ"]
    G3["STA<br/>λ, k<sub>1</sub>, k<sub>2</sub>"]
    G4["LADRC<br/>ω<sub>o</sub>, b<sub>0</sub>, ESO"]

    H(["Inicia loop principal"])

    A --> B --> C --> D
    D -->|Sim| E --> G
    D -->|Não| F --> G
    G -->|CTC| G1 --> H
    G -->|SMC| G2 --> H
    G -->|STA| G3 --> H
    G -->|LADRC| G4 --> H
```

---

## Diagrama 7b — Loop de Simulação: Passo de Controle

```mermaid
flowchart TD
    A(["Para cada substep $$dt_{phys}$$"])

    B["Gera referência cartesiana<br/>P<sub>ref</sub>, V<sub>ref</sub>, A<sub>ref</sub>"]
    C["CLIK — Converte para juntas<br/>q<sub>d</sub>, q̇<sub>d</sub>, q̈<sub>d</sub>"]
    D["Avalia modelo numericamente<br/>M(q), C(q, q̇), G(q)"]
    E["Calcula erros<br/>e = q<sub>d</sub> − q,   ė = q̇<sub>d</sub> − q̇"]

    F{Controlador?}
    F1["CTC-PID<br/>τ = M(q̈<sub>d</sub> + K<sub>D</sub>ė + K<sub>P</sub>e) + C + G"]
    F2["CT-SMC<br/>τ = M(q̈<sub>f</sub> − λė − K·tanh(S/φ)) + C + G"]
    F3["STA<br/>τ = M(q̈<sub>f</sub> − λė + v<sub>sw</sub>) + C + G"]
    F4["LADRC<br/>τ = (u<sub>0</sub> − z<sub>3</sub>) / b<sub>0</sub> + G + C"]

    G["Adiciona perturbação externa"]
    H["Subtrai forças passivas<br/>atrito + arrasto hidrodinâmico"]
    I["Integra dinâmica direta<br/>q̈ = M<sup>-1</sup>(τ<sub>total</sub> − C − G − τ<sub>pass</sub>)"]
    J["Atualiza estado<br/>q += q̇·dt,   q̇ += q̈·dt"]

    K{Estado<br/>finito?}
    L(["Erro numérico — interrompe"])
    M(["Salva resultado e animação"])

    A --> B --> C --> D --> E --> F
    F -->|CTC| F1 --> G
    F -->|SMC| F2 --> G
    F -->|STA| F3 --> G
    F -->|LADRC| F4 --> G
    G --> H --> I --> J --> K
    K -->|Não| L
    K -->|Sim| M
```

---

## Diagrama 8 — Controlador CTC-PID

```mermaid
flowchart LR
    REF(["Referências<br/>q<sub>d</sub>, q̇<sub>d</sub>, q̈<sub>d</sub>"])
    MEAS(["Estado medido<br/>q, q̇"])
    MODEL(["Modelo dinâmico<br/>M, C, G"])

    subgraph CTC["Torque Computado + PID"]
        E1["Calcula erros<br/>e = q<sub>d</sub> − q,   ė = q̇<sub>d</sub> − q̇"]
        E2["Acumula integral<br/>com anti-windup"]
        E3["Sinal auxiliar<br/>v = q̈<sub>d</sub> + K<sub>D</sub>ė + K<sub>P</sub>e + K<sub>I</sub>∫e"]
        E4["Torque final<br/>τ = Mv + C + G"]
        E1 --> E2 --> E3 --> E4
    end

    PLANT(["Planta<br/>Mq̈ + Cq̇ + G = τ"])

    REF --> E1
    MEAS --> E1
    MODEL --> E4
    E4 --> PLANT
    PLANT -->|"q, q̇"| MEAS
```

---

## Diagrama 9 — Controlador CT-SMC

```mermaid
flowchart TD
    A(["Entradas: $$q,\; \dot{q},\; q_d,\; \dot{q}_d,\; \ddot{q}_d,\; M,\; C,\; G$$"])

    B["Filtra aceleração de referência<br/>q̈<sub>f</sub> = α·q̈<sub>d</sub> + (1−α)·q̈<sub>f_prev</sub>"]
    C["Define superfície deslizante<br/>S = ė + λe"]
    D["Lei de chaveamento suavizada<br/>v = q̈<sub>f</sub> − λė − K·tanh(S/φ)"]
    E["Torque final<br/>τ = Mv + C + G"]

    F{K > perturbação<br/>máxima?}
    G["Estado converge para camada limite<br/>|S| ≤ Kφ / (K − D)"]
    H["Sem garantia de estabilidade<br/>Aumentar K"]

    A --> B --> C --> D --> E --> F
    F -->|Sim| G
    F -->|Não| H
```

---

## Diagrama 10 — Controlador STA (Super-Twisting)

```mermaid
flowchart TD
    A(["Entradas: $$q,\; \dot{q},\; q_d,\; \dot{q}_d,\; \ddot{q}_d,\; dt,\; M,\; C,\; G$$"])

    B["Filtra aceleração de referência<br/>q̈<sub>f</sub> = α·q̈<sub>d</sub> + (1−α)·q̈<sub>f_prev</sub>"]
    C["Define superfície deslizante<br/>S = ė + λe"]
    D["Termo de chaveamento<br/>v<sub>sw</sub> = −k<sub>1</sub>|S|<sup>½</sup>·sign(S) + z"]
    E["Integra estado interno<br/>z −= k<sub>2</sub>·sign(S)·dt"]
    F["Torque final<br/>τ = M(q̈<sub>f</sub> − λė + v<sub>sw</sub>) + C + G"]

    G["Convergência em tempo finito se<br/>k<sub>2</sub> > δ  e  k<sub>1</sub> > 2√(k<sub>2</sub>·δ)"]

    A --> B --> C --> D --> E --> F
    F -.->|"Moreno & Osorio, 2012"| G
```

---

## Diagrama 11a — LADRC: Observador de Estado Estendido

```mermaid
flowchart TD
    A(["Entrada: $$q_{med},\; \tau_{prev}$$"])

    B["Parâmetros do ESO<br/>β<sub>1</sub> = 3ω<sub>o</sub>,  β<sub>2</sub> = 3ω<sub>o</sub><sup>2</sup>,  β<sub>3</sub> = ω<sub>o</sub><sup>3</sup>"]
    C["Calcula erro de observação<br/>ε = z<sub>1</sub> − q<sub>med</sub>"]
    D["Atualiza estados do ESO<br/>z<sub>1</sub> ≈ q,  z<sub>2</sub> ≈ q̇,  z<sub>3</sub> ≈ distúrbio total"]
    E["Satura estados<br/>para evitar divergência"]
    F["Filtra z<sub>3</sub> para reduzir<br/>ruído de alta frequência"]

    G{"$$\omega_o \cdot dt \leq 0.1$$?"}
    H["ESO estável numericamente"]
    I["Risco de instabilidade<br/>Reduzir ω<sub>o</sub>"]

    A --> B --> C --> D --> E --> F --> G
    G -->|Sim| H
    G -->|Não| I
```

---

## Diagrama 11b — LADRC: Lei de Controle

```mermaid
flowchart TD
    A(["z<sub>3</sub> filtrado do ESO<br/>q<sub>d</sub>, q̇<sub>d</sub>, q̈<sub>d</sub>, q, q̇"])

    B{$$b_0$$ automático?}
    B1["b<sub>0</sub> = 1 / M<sub>ii</sub>(q<sub>0</sub>)<br/>Estimativa pela inércia inicial"]
    B2["b<sub>0</sub> definido<br/>pelo usuário"]

    C["Sinal PD sobre a referência<br/>u<sub>0</sub> = q̈<sub>d</sub> + k<sub>p</sub>e + k<sub>d</sub>ė"]
    D["Cancela distúrbio estimado<br/>u<sub>adrc</sub> = (u<sub>0</sub> − z<sub>3</sub>) / b<sub>0</sub>"]
    E["Suaviza saída<br/>por filtro de torque"]

    F{Feedforward<br/>de G e C?}
    G["τ = u<sub>adrc</sub> + G + C<br/>Cancela não-linearidades conhecidas"]
    H["τ = u<sub>adrc</sub><br/>Distúrbio absorvido só pelo ESO"]

    B -->|Sim| B1 --> C
    B -->|Não| B2 --> C
    A --> C --> D --> E --> F
    F -->|Sim| G
    F -->|Não| H
```

---

## Diagrama 12 — Paralelismo em Dois Níveis

```mermaid
flowchart TD
    subgraph N1["Nível 1 — Derivação Simbólica  (executa uma vez)"]
        A1["$$M(q)$$ disponível"]
        B1{Múltiplos<br/>cores?}
        C1["Diferencia M em relação<br/>a cada q<sub>k</sub> em paralelo"]
        D1["Diferencia em série<br/>(modo executável)"]
        E1["Monta C(q, q̇)<br/>pelos símbolos de Christoffel"]
        A1 --> B1
        B1 -->|Sim| C1 --> E1
        B1 -->|Não| D1 --> E1
    end

    subgraph N2["Nível 2 — Avaliação Numérica  (a cada dt_phys)"]
        A2["Executor persistente<br/>pré-aquecido com lambdify"]
        B2{Paralelismo<br/>ativado?}
        C2["Avalia M, C, G<br/>em 3 processos simultâneos"]
        D2["Avalia M, C, G<br/>em sequência"]
        E2(["M, C, G prontos<br/>para o controlador"])
        A2 --> B2
        B2 -->|Sim| C2 --> E2
        B2 -->|Não| D2 --> E2
    end

    E1 --> COMP["Compila com lambdify + CSE<br/>(executa uma vez)"]
    COMP --> A2
```

---

*Gerado a partir da análise de `engine.py`, `simulator.py` e dos capítulos do TCC — Hephaestus.*
