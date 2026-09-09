# Mini surrogate project

Surrogate model (PyTorch) del esquema numérico 2D FDTD/FD-FV de propagación no lineal de ultrasonidos en medios viscoelásticos con burbujas (onda + Rayleigh-Plesset + Kelvin-Voigt), desarrollado como parte de la tesis en el grupo NANLA (URJC).

Objetivo: entrenar una red que aproxime la salida del solver MATLAB a partir de los parámetros físicos de entrada, como pieza de portfolio que conecta modelado numérico de EDPs con machine learning aplicado a ciencia (surrogate modeling / ML para simulación).

## Estructura

```
data/
  raw/         # salidas del solver MATLAB (.mat / .csv), no versionadas en git
  processed/   # datasets ya limpios/listos para entrenar (tampoco versionados si pesan)
src/
  generate_dataset.py   # lee las salidas de MATLAB y construye el dataset de entrenamiento
  model.py               # arquitectura del surrogate en PyTorch
  train.py                # bucle de entrenamiento
  evaluate.py             # métricas y comparación frente al solver
notebooks/    # exploración de datos y resultados
tests/         # tests del código en src/
```

## Estado

- [ ] Definir el espacio de parámetros a barrer en MATLAB (Fase 0 del plan)
- [ ] Generar el dataset sintético (Fase 1)
- [ ] Implementar y entrenar el modelo v1 (magnitud reducida) (Fase 2)
- [ ] Validar y analizar resultados (Fase 3)
- [ ] Pulir README y publicar el repo (Fase 4)

Ver el plan completo en el proyecto de Claude ("Profesional" → `mini-proyecto-surrogate-modeling-elena.md`).

## Setup

```
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell en Windows
pip install -r requirements.txt
```
