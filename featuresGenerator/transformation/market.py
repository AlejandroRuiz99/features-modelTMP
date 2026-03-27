"""
Market Adjust — capa de ajuste por cuotas de mercado.

Las cuotas se usan como señal de calibración externa, no como núcleo
del sistema. Se mantiene trazabilidad total entre expectativa estadística
y señal de mercado, con un score de alineación explícito.
"""

from __future__ import annotations

import math
from typing import Optional

from core.config import MARKET_ALIGNMENT_THRESHOLD

_ALIGNMENT_TIERS: list[tuple[float, str]] = [
    (0.80,                      "Alta alineación modelo-mercado — expectativas consistentes"),
    (MARKET_ALIGNMENT_THRESHOLD,"Discrepancia moderada — revisar supuestos de forma o contexto"),
    (0.00,                      "Divergencia significativa — señal de valor potencial o dato faltante"),
]

_GLOBAL_ALIGNMENT_TIERS: list[tuple[float, str]] = [
    (0.80,                      "Alta alineación global modelo-mercado"),
    (MARKET_ALIGNMENT_THRESHOLD,"Alineación global moderada"),
    (0.00,                      "Divergencia global relevante"),
]

_TENDENCIA_THRESHOLD = 0.08


# ── Probabilidades implícitas ─────────────────────────────────────────────────

def implied_probabilities(
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> dict:
    """
    Convierte cuotas decimales 1X2 a probabilidades sin overround.

    El overround (margen de la casa) se elimina normalizando las
    probabilidades brutas a que sumen 1.0.
    """
    if any(o <= 1.0 for o in (odds_h, odds_d, odds_a)):
        raise ValueError("Las cuotas deben ser mayores que 1.0")

    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a
    overround = total - 1.0

    return {
        "prob_local": round(raw_h / total, 4),
        "prob_draw": round(raw_d / total, 4),
        "prob_visitante": round(raw_a / total, 4),
        "overround": round(overround, 4),
        "odds_raw": {"local": odds_h, "draw": odds_d, "visitante": odds_a},
    }


def implied_prob_ou(
    odds_over: float,
    odds_under: Optional[float] = None,
) -> dict:
    """
    Convierte cuotas Over/Under a probabilidades sin overround.
    Si solo se provee odds_over, se asume complementaria (sin overround).
    """
    if odds_over <= 1.0:
        raise ValueError("La cuota Over debe ser mayor que 1.0")

    raw_o = 1.0 / odds_over
    if odds_under is not None and odds_under > 1.0:
        raw_u = 1.0 / odds_under
        total = raw_o + raw_u
        return {
            "prob_over": round(raw_o / total, 4),
            "prob_under": round(raw_u / total, 4),
            "overround": round(total - 1.0, 4),
        }
    else:
        # Solo cuota over: complementaria directa (sin overround explícito)
        return {
            "prob_over": round(raw_o, 4),
            "prob_under": round(1.0 - raw_o, 4),
            "overround": None,
        }


# ── Score de alineación modelo-mercado ────────────────────────────────────────

def market_alignment_score(
    prob_stat: dict,
    prob_market: dict,
) -> dict:
    """
    Compara probabilidades estadísticas internas con las implícitas del mercado.

    prob_stat: xgoals dict (prob_local_win, prob_draw, prob_visitante_win)
    prob_market: implied_probabilities() dict (prob_local, prob_draw, prob_visitante)

    Retorna diferencias por resultado y un alignment_score global [0, 1],
    donde 1.0 indica alineación perfecta.
    """
    diff_l = prob_stat["prob_local_win"] - prob_market["prob_local"]
    diff_d = prob_stat["prob_draw"] - prob_market["prob_draw"]
    diff_v = prob_stat["prob_visitante_win"] - prob_market["prob_visitante"]

    # Divergencia total (suma de absolutos / 2, normalizado)
    divergence = (abs(diff_l) + abs(diff_d) + abs(diff_v)) / 2.0
    alignment = max(0.0, 1.0 - divergence * 2.0)

    def _tendencia(diff: float) -> str:
        if diff >  _TENDENCIA_THRESHOLD: return "modelo_asigna_mas_probabilidad"
        if diff < -_TENDENCIA_THRESHOLD: return "mercado_asigna_mas_probabilidad"
        return "alineado"

    interpretacion = next(t for thresh, t in _ALIGNMENT_TIERS if alignment >= thresh)

    return {
        "alignment_score": round(alignment, 3),
        "divergencia_total": round(divergence, 4),
        "diferencias": {
            "local": round(diff_l, 4),
            "draw": round(diff_d, 4),
            "visitante": round(diff_v, 4),
        },
        "tendencias": {
            "local": _tendencia(diff_l),
            "draw": _tendencia(diff_d),
            "visitante": _tendencia(diff_v),
        },
        "interpretacion": interpretacion,
    }


# ── Integración en el knowledge pack ─────────────────────────────────────────

def _sigmoid_over_prob(model_total: float, line: float, scale: float) -> float:
    x = (model_total - line) / max(0.001, scale)
    return 1.0 / (1.0 + math.exp(-x))


def _market_ou_block(
    *,
    market_name: str,
    line: Optional[float],
    odds_over: Optional[float],
    odds_under: Optional[float],
    model_total: Optional[float],
    scale: float,
) -> Optional[dict]:
    if odds_over is None:
        return None
    try:
        ou_probs = implied_prob_ou(odds_over, odds_under)
    except ValueError:
        return None

    block = {
        "market": market_name,
        "linea": line,
        "cuota_over": odds_over,
        "cuota_under": odds_under,
        "prob_mercado_over": ou_probs["prob_over"],
        "prob_mercado_under": ou_probs["prob_under"],
        "overround": ou_probs["overround"],
        "model_total": model_total,
    }
    if model_total is not None and line is not None:
        p_model_over = _sigmoid_over_prob(model_total, line, scale)
        diff = p_model_over - ou_probs["prob_over"]
        block.update(
            {
                "prob_modelo_over": round(p_model_over, 4),
                "prob_modelo_under": round(1.0 - p_model_over, 4),
                "diferencia_over": round(diff, 4),
                "interpretacion": (
                    "modelo por encima de mercado en el over"
                    if diff > 0.07
                    else "mercado por encima del modelo en el over"
                    if diff < -0.07
                    else "alineado"
                ),
                "alignment_score": round(max(0.0, 1.0 - min(1.0, abs(diff) * 2.0)), 3),
            }
        )
    else:
        block.update(
            {
                "prob_modelo_over": None,
                "prob_modelo_under": None,
                "diferencia_over": None,
                "interpretacion": "sin señal interna suficiente para comparar",
                "alignment_score": None,
            }
        )
    return block


def ajustar_knowledge_pack(
    knowledge_pack: dict,
    market_input: Optional[dict] = None,
) -> dict:
    """
    Añade la capa de señal de mercado al knowledge pack con trazabilidad completa.

    market_input es un dict unificado con todas las cuotas prepartido:
      {"1x2": {"local": ..., "empate": ..., "visitante": ...},
       "goals_ou": {"line": 2.5, "over": ..., "under": ...},
       "fouls_ou": ..., "corners_ou": ..., ...}

    Nunca modifica las metricas estadisticas base — solo añade la capa de mercado.
    """
    if not market_input:
        return knowledge_pack

    stat_probs = knowledge_pack["expected_metrics"]["xgoals"]
    expected = knowledge_pack.get("expected_metrics", {})
    xvol = expected.get("xvolumen_eventos", {})
    xf = expected.get("xfouls", {})
    xt = expected.get("xtarjetas", {})

    market_signal: dict = {
        "version": "unified_market_v2",
        "markets": {},
        "available_markets": [],
    }

    alignments: list[float] = []

    # 1X2
    one_x_two = market_input.get("1x2", {})
    if all(one_x_two.get(k) is not None for k in ("local", "empate", "visitante")):
        try:
            market_probs = implied_probabilities(
                one_x_two["local"], one_x_two["empate"], one_x_two["visitante"]
            )
            alignment = market_alignment_score(stat_probs, market_probs)
            market_signal["markets"]["1x2"] = {
                "probabilidades_mercado": market_probs,
                "probabilidades_modelo": {
                    "prob_local":     stat_probs["prob_local_win"],
                    "prob_draw":      stat_probs["prob_draw"],
                    "prob_visitante": stat_probs["prob_visitante_win"],
                },
                "alignment": alignment,
                "cuotas":    one_x_two,
            }
            market_signal["available_markets"].append("1x2")
            alignments.append(alignment["alignment_score"])
        except ValueError:
            pass

    # Over/Under goals
    goals_ou = market_input.get("goals_ou", {})
    if goals_ou:
        b = _market_ou_block(
            market_name="goals_ou",
            line=goals_ou.get("line", 2.5),
            odds_over=goals_ou.get("over"),
            odds_under=goals_ou.get("under"),
            model_total=stat_probs.get("xg_total"),
            scale=0.9,
        )
        if b:
            market_signal["markets"]["goals_ou"] = b
            market_signal["available_markets"].append("goals_ou")
            if b.get("alignment_score") is not None:
                alignments.append(b["alignment_score"])

    # Mercados micro en un único sitio.
    specs = [
        ("fouls_ou", "fouls_ou", xf.get("total"), 2.8),
        ("cards_ou", "cards_ou", xt.get("xtarjetas_total"), 1.2),
        ("corners_ou", "corners_ou", xvol.get("corners_total"), 1.8),
        ("shots_ou", "shots_ou", xvol.get("shots_total"), 2.5),
        ("shots_on_target_ou", "shots_on_target_ou", xvol.get("shots_on_target_total"), 1.4),
        ("offsides_ou", "offsides_ou", xvol.get("offsides_total"), 1.0),
    ]

    for key, name, model_total, scale in specs:
        mk = market_input.get(key, {})
        if not mk:
            continue
        b = _market_ou_block(
            market_name=name,
            line=mk.get("line"),
            odds_over=mk.get("over"),
            odds_under=mk.get("under"),
            model_total=model_total,
            scale=scale,
        )
        if b:
            market_signal["markets"][name] = b
            market_signal["available_markets"].append(name)
            if b.get("alignment_score") is not None:
                alignments.append(b["alignment_score"])

    if alignments:
        global_alignment = round(sum(alignments) / len(alignments), 3)
        market_signal["global_alignment"] = {
            "score":         global_alignment,
            "n_markets":     len(alignments),
            "interpretacion": next(t for thresh, t in _GLOBAL_ALIGNMENT_TIERS if global_alignment >= thresh),
        }

    if not market_signal["available_markets"]:
        return knowledge_pack

    knowledge_pack["market_signal"] = market_signal
    return knowledge_pack
