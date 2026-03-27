"""
evaluation/ — Validacion y calibracion de las metricas heuristicas del featuresGenerator.

Estructura:
  base.py             Metricas estadisticas compartidas (MAE, RMSE, Pearson, Brier, log-loss)
  xfouls_eval.py      Walk-forward + calibracion de DECAY_LAMBDA y ALPHA_CARD_PRESSURE
  xgoals_eval.py      Backtesting de xGoals vs goles reales
  xposesion_eval.py   Backtesting de xPosesion vs posesion real
  referee_eval.py     Valor anadido del factor arbitro en la prediccion de faltas
  iap_eval.py         Validacion de los pesos IAP como predictores de faltas
  aggressivity_eval.py  Calibracion de agresividad por volumen
  run_all.py          Orquestador: ejecuta todos los evaluadores y muestra resumen

Uso rapido (desde featuresGenerator/):
  python -m evaluation.run_all
  python -m evaluation.run_all --update-config
  python -m evaluation.xfouls_eval
"""
