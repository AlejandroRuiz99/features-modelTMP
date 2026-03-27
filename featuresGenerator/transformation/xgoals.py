"""xGoals — Goles esperados y probabilidades de resultado via Poisson bivariate."""

from __future__ import annotations

import math

from core.config import HOME_GOALS_FACTOR
from core.helpers import safe

AWAY_GOALS_FACTOR = round(2.0 - HOME_GOALS_FACTOR, 4)


def _poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_outcome_probs(xg_local: float, xg_vis: float) -> dict:
    """Probabilidades de resultado 1X2 + over 2.5 + BTTS via convolucion Poisson."""
    xg_total = xg_local + xg_vis
    p_over25 = 1.0 - sum(_poisson_prob(xg_total, k) for k in range(3))
    p_btts = (1.0 - _poisson_prob(xg_local, 0)) * (1.0 - _poisson_prob(xg_vis, 0))

    p_local_win = p_draw = p_vis_win = 0.0
    for i in range(9):
        for j in range(9):
            p = _poisson_prob(xg_local, i) * _poisson_prob(xg_vis, j)
            if   i > j: p_local_win += p
            elif i == j: p_draw      += p
            else:        p_vis_win   += p

    total = p_local_win + p_draw + p_vis_win
    if total > 0:
        p_local_win /= total; p_draw /= total; p_vis_win /= total

    return {
        "prob_over25":        round(p_over25, 3),
        "prob_under25":       round(1.0 - p_over25, 3),
        "prob_btts":          round(p_btts, 3),
        "prob_local_win":     round(p_local_win, 3),
        "prob_draw":          round(p_draw, 3),
        "prob_visitante_win": round(p_vis_win, 3),
    }


def calcular_xgoals(xstyles: dict, equipo_local: str, equipo_visitante: str) -> dict:
    """Estima goles esperados combinando tasa ofensiva y defensiva con ventaja local."""
    sl = xstyles.get(equipo_local,     {})
    sv = xstyles.get(equipo_visitante, {})

    xg_local     = ((safe(sl.get("goles"), 1.2) + safe(sv.get("goles_conc"), 1.2)) / 2) * HOME_GOALS_FACTOR
    xg_visitante = ((safe(sv.get("goles"), 1.0) + safe(sl.get("goles_conc"), 1.2)) / 2) * AWAY_GOALS_FACTOR

    return {
        "xg_local":     round(xg_local,     2),
        "xg_visitante": round(xg_visitante, 2),
        "xg_total":     round(xg_local + xg_visitante, 2),
        **poisson_outcome_probs(xg_local, xg_visitante),
    }
