"""
base.py — Métricas estadísticas compartidas para todos los evaluadores.
"""
from __future__ import annotations

import argparse
import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Métricas de regresión
# ---------------------------------------------------------------------------

def mae(preds: Sequence[float], reals: Sequence[float]) -> float:
    """Mean Absolute Error."""
    if not preds:
        return float("nan")
    return sum(abs(p - r) for p, r in zip(preds, reals)) / len(preds)


def rmse(preds: Sequence[float], reals: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    if not preds:
        return float("nan")
    return math.sqrt(sum((p - r) ** 2 for p, r in zip(preds, reals)) / len(preds))


def bias(preds: Sequence[float], reals: Sequence[float]) -> float:
    """Sesgo medio (positivo = el modelo sobreestima)."""
    if not preds:
        return float("nan")
    return sum(p - r for p, r in zip(preds, reals)) / len(preds)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Correlación de Pearson entre predicciones y valores reales."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Métricas de clasificación (probabilidades)
# ---------------------------------------------------------------------------

def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Log-loss binaria. Menor es mejor; baseline ~0.693 (p=0.5 siempre)."""
    if not probs:
        return float("nan")
    eps = 1e-9
    return -sum(
        y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))
        for p, y in zip(probs, outcomes)
    ) / len(probs)


def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Brier score. 0 = perfecto, 0.25 = sin información (p=0.5 siempre)."""
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def brier_skill(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """
    Brier Skill Score (BSS): cuánto mejor es el modelo respecto a predecir
    siempre la frecuencia base.  BSS > 0 = el modelo añade valor.
    """
    if not outcomes:
        return float("nan")
    base_rate = sum(outcomes) / len(outcomes)
    bs_model = brier(probs, outcomes)
    bs_clim = brier([base_rate] * len(outcomes), outcomes)
    return 1.0 - bs_model / bs_clim if bs_clim > 0 else float("nan")


# ---------------------------------------------------------------------------
# Utilidades de impresión
# ---------------------------------------------------------------------------

_W = 58


def header(title: str) -> None:
    print(f"\n{'=' * _W}")
    print(f"  {title}")
    print(f"{'=' * _W}")


def section(title: str) -> None:
    print(f"\n  {'-' * (_W - 4)}")
    print(f"  {title}")
    print(f"  {'-' * (_W - 4)}")


def row(label: str, value: str) -> None:
    print(f"    {label:<28} {value}")


def footer() -> None:
    print(f"{'=' * _W}\n")


# ---------------------------------------------------------------------------
# CLI helpers (evitar boilerplate duplicado en cada evaluador)
# ---------------------------------------------------------------------------

def load_partidos() -> list[dict]:
    """Carga partidos de Supabase para uso en evaluadores standalone."""
    from selection import supabase_client

    print("[INFO] Cargando partidos desde Supabase...")
    partidos = supabase_client.fetch_all_matches()
    if not partidos:
        raise SystemExit("ERROR: No hay partidos disponibles.")
    print(f"[INFO] {len(partidos)} partidos cargados.")
    return partidos


def eval_cli(description: str, *, calibrable: bool = False) -> argparse.Namespace:
    """Parser estándar para CLIs de evaluación."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n", type=int, default=150, help="Nº de partidos recientes (default: 150)")
    if calibrable:
        parser.add_argument("--calibrar", action="store_true", help="Grid search de calibración")
        parser.add_argument("--update-config", action="store_true", help="Actualizar config.yaml con params óptimos")
    return parser.parse_args()
