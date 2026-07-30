"""Detección de Expected Value (EV) para mercados Over/Under de faltas.

Dado P(over L) del modelo y las cuotas decimales del mercado, calcula:
  - edge    : ventaja del modelo vs mercado (sin vig)
  - kelly   : fracción de bankroll recomendada (Kelly fraccional al 25%)
  - bet     : dirección recomendada ("over" | "under" | None)

Ejemplo:
    result = compute_ev(
        line=25.5,
        p_over_model=0.62,
        odds_over=1.85,
        odds_under=2.00,
        kelly_fraction=0.25,
        min_edge=0.03,
    )
    # result = {"line": 25.5, "bet": "over", "edge": 0.081, "kelly_stake": 0.042, ...}

kelly_scale (overlay P4):
    Optional multiplier [0.25, 1.0] applied to the Kelly fraction.
    Default 1.0 → backward-compatible (no change).
    Output includes both 'kelly_raw' (pre-scale) and 'kelly_scaled' (post-scale).
    'kelly_stake' equals 'kelly_scaled' for call-site compatibility.
"""

from __future__ import annotations

from typing import TypedDict


class EVResult(TypedDict):
    line: float
    bet: str  # "over" | "under"
    edge: float  # ventaja neta (predicted_p - no_vig_p)
    no_vig_p: float  # probabilidad sin margen de la casa
    kelly_stake: float  # fracción de bankroll (Kelly x kelly_fraction x kelly_scale)
    odds: float  # cuota decimal de la apuesta recomendada
    expected_return: float  # EV esperado por unidad apostada
    kelly_raw: float  # Kelly stake BEFORE kelly_scale is applied
    kelly_scaled: float  # Kelly stake AFTER kelly_scale is applied (= kelly_stake)


def _no_vig_prob(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Elimina el margen de la casa y devuelve (p_a_novig, p_b_novig)."""
    if odds_a <= 1.0 or odds_b <= 1.0:
        return 0.5, 0.5
    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def compute_ev(
    line: float,
    p_over_model: float,
    odds_over: float,
    odds_under: float,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    kelly_scale: float = 1.0,
) -> EVResult | None:
    """Calcula EV para una línea OU.

    Args:
        line: línea OU (p.ej. 25.5).
        p_over_model: probabilidad del modelo de que faltas > line.
        odds_over: cuota decimal del mercado para over.
        odds_under: cuota decimal del mercado para under.
        kelly_fraction: fracción del Kelly completo (0.25 = Kelly 25%).
        min_edge: edge mínimo para considerar apuesta (0.03 = 3%).
        kelly_scale: overlay multiplier [0.25, 1.0]. Default 1.0 is a no-op.
            Multiplied into the final Kelly fraction. Output reports both
            'kelly_raw' (before scale) and 'kelly_scaled' (after scale).

    Returns:
        EVResult si hay edge suficiente, None si no hay valor.
    """
    if not (0.0 < p_over_model < 1.0):
        return None
    if odds_over <= 1.0 or odds_under <= 1.0:
        return None

    no_vig_over, no_vig_under = _no_vig_prob(odds_over, odds_under)
    p_under_model = 1.0 - p_over_model

    edge_over = p_over_model - no_vig_over
    edge_under = p_under_model - no_vig_under

    # Determinar lado con mayor ventaja
    if edge_over >= edge_under:
        edge = edge_over
        bet = "over"
        p_model = p_over_model
        no_vig_p = no_vig_over
        odds = odds_over
    else:
        edge = edge_under
        bet = "under"
        p_model = p_under_model
        no_vig_p = no_vig_under
        odds = odds_under

    if edge < min_edge:
        return None

    # Kelly fraccional: f = kelly_fraction x edge / (odds - 1)
    # (version discreta: ganancia neta = odds - 1 si gana, -1 si pierde)
    net_odds = odds - 1.0
    kelly_unrounded = kelly_fraction * edge / net_odds if net_odds > 0 else 0.0
    kelly_unrounded = min(kelly_unrounded, 0.20)  # cap en 20% del bankroll

    # Apply overlay kelly_scale (P4) BEFORE rounding so kelly_scaled = kelly_raw * scale exactly
    kelly_raw = round(kelly_unrounded, 4)
    kelly_scaled = round(kelly_unrounded * kelly_scale, 4)

    expected_return = round(p_model * net_odds - (1.0 - p_model), 4)

    return EVResult(
        line=line,
        bet=bet,
        edge=round(edge, 4),
        no_vig_p=round(no_vig_p, 4),
        kelly_stake=kelly_scaled,  # backward compat: kelly_stake = kelly_scaled
        odds=odds,
        expected_return=expected_return,
        kelly_raw=kelly_raw,
        kelly_scaled=kelly_scaled,
    )


def compute_ev_all_lines(
    ou_table: dict[float, tuple[float, float]],
    market_odds: dict[float, tuple[float, float]],
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    kelly_scale: float = 1.0,
) -> list[EVResult]:
    """Calcula EV para todas las líneas con cuotas disponibles.

    Args:
        ou_table: {line: (p_over, p_under)} del modelo (calibrado o no).
        market_odds: {line: (odds_over, odds_under)} del mercado.
        kelly_scale: overlay multiplier [0.25, 1.0]. Default 1.0 is a no-op.

    Returns:
        Lista de EVResult ordenada por edge descendente.
    """
    results = []
    for line, (p_over, _) in ou_table.items():
        if line not in market_odds:
            continue
        odds_over, odds_under = market_odds[line]
        ev = compute_ev(
            line,
            p_over,
            odds_over,
            odds_under,
            kelly_fraction,
            min_edge,
            kelly_scale=kelly_scale,
        )
        if ev is not None:
            results.append(ev)
    return sorted(results, key=lambda r: r["edge"], reverse=True)
