"""Infraestructura compartida del pipeline KDD.

Modulos:
  config       — constantes y secretos (desde config.yaml + .env)
  helpers      — funciones matematicas puras (decay_weight, parse_date, safe)
  utils        — normalización de texto, alias de equipos, utilidades de mercado
  state_cache  — estado estadistico precalculado (singleton thread-safe)
"""
