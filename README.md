# RobotModelPython

## Base móvel e fallback
Quando `has_vehicle_base` está desativado, o solver ignora lógica de base móvel e
opera como robô fixo. Se a base móvel estiver ativa sem índices explícitos, o
fallback assume os primeiros 6 DOFs como base e o restante como manipulador.
