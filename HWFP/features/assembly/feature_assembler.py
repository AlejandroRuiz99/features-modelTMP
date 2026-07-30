"""Ensamblador de features para prediccion de faltas.

Pipeline:
  1. _resolve_inputs         — resuelve nombres de equipos y arbitro
  2. Calendario multi-competicion via cal_index (Supabase liga_calendar)
  3. _compute_expected_stats — xFouls + knowledge pack (doble pasada con ICC)
  4. _fetch_and_apply_market — cuotas de mercado y ajuste de probabilidades
  5. _assemble_contract      — ensambla el dict anidado completo
  6. _flatten                — convierte a flat dict para el modelo

Dos funciones publicas:
  build_features  — flat dict listo para el ensemble (API principal)
  build_detailed  — dict anidado con datos completos (para reports/debug)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from HWFP.features.transformation import (
    ajustar_knowledge_pack,
    buscar_arbitro,
    buscar_equipo,
    calcular_perfiles_arbitros,
    calcular_xfouls,
    ensamblar_knowledge_pack,
)
from HWFP.features.transformation.referee_gmm import calcular_perfiles_gmm, get_perfil_gmm_o_default
from HWFP.features.transformation.competitive_context import build_context_payload
from HWFP.features.transformation.match_labels import build_labels_category
from HWFP.features.assembly.betting_odds import build_market_category
from HWFP.features.assembly.completeness import build_quality_category
from HWFP.features.assembly.feature_registry import FeatureRegistry
from HWFP.features.core.utils import no_vig, no_vig_3, market_entropy
from HWFP.features.core.config import DERBIES as _DERBIES

_AGG_SCALE_TEAM = 15.0
_AGG_SCALE_TOTAL = 30.0


# ---------------------------------------------------------------------------
# Paso 1: Resolver inputs
# ---------------------------------------------------------------------------


def _resolve_inputs(
    state: dict,
    equipo_local_input: str,
    equipo_visitante_input: str,
    arbitro_input: str | None,
    jornada: int | None,
) -> tuple[str, str, str | None, int, list[str]]:
    """Resuelve nombres de equipos y arbitro desde el estado historico.

    Returns:
        (eq_local, eq_visit, arbitro_resuelto, jornada_final, warnings)
    """
    scores = state["scores"]
    warnings: list[str] = []

    eq_local = buscar_equipo(equipo_local_input, scores)
    eq_visit = buscar_equipo(equipo_visitante_input, scores)
    if not eq_local:
        raise ValueError(f"Equipo local no encontrado: {equipo_local_input}")
    if not eq_visit:
        raise ValueError(f"Equipo visitante no encontrado: {equipo_visitante_input}")
    if eq_local == eq_visit:
        raise ValueError("Equipo local y visitante no pueden ser el mismo.")

    refs = state.get("ref_perfiles") or calcular_perfiles_arbitros(state["partidos"])
    arbitro_resuelto = buscar_arbitro(arbitro_input, refs) if arbitro_input else None
    if arbitro_input and not arbitro_resuelto:
        warnings.append(f"Arbitro no encontrado en perfiles: {arbitro_input}.")
        arbitro_resuelto = arbitro_input

    return eq_local, eq_visit, arbitro_resuelto, jornada or 1, warnings


# ---------------------------------------------------------------------------
# Paso 2: Calcular estadisticas esperadas (doble pasada)
# ---------------------------------------------------------------------------


def _ctx_payload_to_comp(ctx_payload: dict) -> dict:
    """Convierte el payload de competitive_context al formato que espera knowledge.py.

    Extrae factores (xg_factor, xfouls_factor, posesion_delta_pp) e ICC ya calculados,
    evitando cualquier recalculo en knowledge.py.
    """

    def _side(data: dict) -> dict:
        compet = data.get("competitividad", {})
        icc = compet.get("icc_score")
        factors = compet.get(
            "factors",
            {"xg_factor": 1.0, "xfouls_factor": 1.0, "posesion_delta_pp": 0.0},
        )
        lectura = (
            "foco competitivo alto en liga"
            if icc is not None and icc >= 0.20
            else "foco de liga condicionado por otras prioridades"
            if icc is not None and icc <= -0.20
            else "foco liguero neutro"
        )
        return {"factors": factors, "scores": {"icc": icc}, "lectura": lectura}

    competitivo = ctx_payload.get("competitivo", {})
    return {
        "local": _side(competitivo.get("local", {})),
        "visitante": _side(competitivo.get("visitante", {})),
    }


def _compute_expected_stats(
    state: dict,
    eq_local: str,
    eq_visit: str,
    arbitro: str | None,
    jornada: int,
    arbitraje_source: str,
    fecha_partido: str | None,
) -> tuple[dict, dict, dict]:
    """Calcula knowledge pack y contexto competitivo con doble pasada.

    La doble pasada es necesaria porque:
      1. El knowledge pack necesita ICC para ajustar xFouls/xGoals.
      2. El contexto competitivo (ICC) se calcula con el contexto de temporada del kp.

    El calendario multi-competicion se obtiene de cal_index (Supabase liga_calendar).

    Returns:
        (knowledge_pack, ctx_payload, xfouls_result)
    """
    partidos = state["partidos"]
    xstyles = state["xstyles"]
    refs = state.get("ref_perfiles") or calcular_perfiles_arbitros(partidos)
    xf = calcular_xfouls(partidos, eq_local, eq_visit, arbitro)

    kp_kwargs = dict(
        equipo_local=eq_local,
        equipo_visitante=eq_visit,
        partidos=partidos,
        xstyles=xstyles,
        xfouls_result=xf,
        ref_perfiles=refs,
        arbitro=arbitro,
        jornada=jornada,
    )
    ctx_kwargs = dict(
        state=state,
        eq_local=eq_local,
        eq_visit=eq_visit,
        jornada=jornada,
        arbitro=arbitro,
        arbitraje_source=arbitraje_source,
        match_date=fecha_partido,
    )

    # Pasada 1: sin ICC → contexto de temporada → calcular ICC real
    kp_initial = ensamblar_knowledge_pack(**kp_kwargs, contexto_comp=None)  # type: ignore[arg-type]
    ctx_payload, _ = build_context_payload(
        **ctx_kwargs,  # type: ignore[arg-type]
        contexto_temporada=kp_initial.get("contexto_temporada", {}),
    )

    # Pasada 2: ICC pre-computado → ajustar metricas
    ctx_comp = _ctx_payload_to_comp(ctx_payload)
    kp_final = ensamblar_knowledge_pack(**kp_kwargs, contexto_comp=ctx_comp)  # type: ignore[arg-type]

    # ctx_payload definitivo usa contexto_temporada del KP final (post-ICC)
    ctx_payload_final, _ = build_context_payload(
        **ctx_kwargs,  # type: ignore[arg-type]
        contexto_temporada=kp_final.get("contexto_temporada", {}),
    )

    return kp_final, ctx_payload_final, xf


# ---------------------------------------------------------------------------
# Paso 4: Fetch y aplicar senal de mercado
# ---------------------------------------------------------------------------


def _fetch_and_apply_market(
    state: dict,
    kp: dict,
    eq_local: str,
    eq_visit: str,
    cuotas_prepartido: dict | None,
    skip_market_fetch: bool,
    warnings: list[str],
    fecha_partido: str | None = None,
) -> tuple[dict, dict | None, list[str], str | None]:
    """Obtiene cuotas, las fusiona con las del usuario, y ajusta el knowledge pack.

    Args:
        fecha_partido: Fecha del partido ('YYYY-MM-DD'). Se propaga a
            `build_market_category` para que el fetch de odds_raw use la
            ventana de scrapes alrededor del partido en vez de solo el
            último scrape global.

    Returns:
        (kp_ajustado, market_input_model, market_used, odds_scraped_at)
    """
    scores = state["scores"]

    if skip_market_fetch:
        market_category = {
            "catalogo_disponible": [],
            "traza": {"odds_scraped_at": None},
        }
        market_input_model = None
        market_used: list[str] = []
        odds_scraped_at = None
    else:
        market_category, market_input_model, market_used, odds_scraped_at = (
            build_market_category(
                scores=scores,
                eq_local=eq_local,
                eq_visit=eq_visit,
                model_market_signal=kp.get("market_signal") or {},
                match_date=fecha_partido,
            )
        )

    if cuotas_prepartido:
        if market_input_model:
            for k, v in cuotas_prepartido.items():
                if isinstance(v, dict) and isinstance(market_input_model.get(k), dict):
                    market_input_model[k] = {**market_input_model[k], **v}
                else:
                    market_input_model[k] = v
        else:
            market_input_model = cuotas_prepartido
            market_used = list(cuotas_prepartido.keys())

    kp_out = kp
    if market_input_model:
        kp_out = ajustar_knowledge_pack(kp, market_input=market_input_model)
        if market_used:
            warnings.append(f"Senal de mercado cargada: {', '.join(market_used)}")
    else:
        warnings.append(
            "No se encontraron cuotas compatibles en odds_raw para este partido."
        )

    return kp_out, market_input_model, market_used, odds_scraped_at


# ---------------------------------------------------------------------------
# Helpers del contrato anidado
# ---------------------------------------------------------------------------


def _h2h_faltas(
    partidos: list[dict], eq_local: str, eq_visit: str
) -> tuple[float | None, int]:
    """Media de faltas totales en enfrentamientos previos entre estos dos equipos."""
    pair = frozenset({eq_local, eq_visit})
    h2h = []
    for p in partidos:
        h = (p.get("home") or {}).get("name") or ""
        a = (p.get("away") or {}).get("name") or ""
        if frozenset({h, a}) == pair:
            f_h = float((p.get("home") or {}).get("fouls") or 0)
            f_a = float((p.get("away") or {}).get("fouls") or 0)
            h2h.append(f_h + f_a)
    if not h2h:
        return None, 0
    return round(sum(h2h) / len(h2h), 2), len(h2h)


def _build_mercado_block(market_input_model: dict[str, Any]) -> dict[str, Any]:
    """Transforma el market_input_model al bloque mercado del contrato."""
    one_x_two = market_input_model.get("1x2", {})
    goals_ou = market_input_model.get("goals_ou", {})
    fouls_ou = market_input_model.get("fouls_ou", {})

    c_local = float(one_x_two.get("local", 2.50) or 2.50)
    c_emp = float(one_x_two.get("empate", 3.30) or 3.30)
    c_vis = float(one_x_two.get("visitante", 2.80) or 2.80)

    c_goles_mas = float(goals_ou.get("over", 1.90) or 1.90)
    c_goles_menos = float(goals_ou.get("under", 1.90) or 1.90)
    linea_goles = float(goals_ou.get("line", 2.5) or 2.5)

    c_faltas_mas = float(fouls_ou.get("over", 1.85) or 1.85)
    c_faltas_menos = float(fouls_ou.get("under", 1.85) or 1.85)
    linea_faltas = float(fouls_ou.get("line", 24.5) or 24.5)

    p_h, p_d, p_a = no_vig_3(c_local, c_emp, c_vis)
    p_goles_over, p_goles_under = no_vig(c_goles_mas, c_goles_menos)
    p_faltas_over, p_faltas_under = no_vig(c_faltas_mas, c_faltas_menos)

    entropia = market_entropy(p_h, p_d, p_a)
    return {
        "resultado": {
            "prob_local": round(p_h, 4),
            "prob_empate": round(p_d, 4),
            "prob_visitante": round(p_a, 4),
            "entropia": round(entropia, 4),
        },
        "goles_ou": {
            "linea": linea_goles,
            "prob_over": round(p_goles_over, 4),
            "prob_under": round(p_goles_under, 4),
        },
        "faltas_total_ou": {
            "linea": linea_faltas,
            "prob_over": round(p_faltas_over, 4),
            "prob_under": round(p_faltas_under, 4),
        },
        "derivadas": {
            "market_entropy": entropia,
            "market_balance": round(1.0 - abs(p_h - p_a), 4),
            "market_favorite_prob": round(max(p_h, p_a), 4),
            "has_market_odds": bool(one_x_two.get("local")),
        },
    }


def _build_team_block(
    *,
    xstyle: dict[str, Any],
    forma_kp: dict[str, Any],
    ctx_payload: dict[str, Any],
) -> dict[str, Any]:
    """Construye el bloque de un equipo para el contrato anidado."""
    tabla = ctx_payload.get("tabla", {})
    forma_ctx = ctx_payload.get("forma_reciente", {})
    calendario = ctx_payload.get("calendario", {})
    compet = ctx_payload.get("competitividad", {})
    factors = compet.get("factors", {})

    return {
        "clasificacion": {
            "posicion": tabla.get("position") or 10,
            "puntos": tabla.get("points"),
            "partidos_jugados": tabla.get("played"),
            "diferencia_goles": tabla.get("goal_diff"),
        },
        "temporada_completa": {
            "faltas_cometidas": round(float(xstyle.get("fouls", 12.0)), 1),
            "faltas_provocadas": round(float(xstyle.get("faltas_prov", 12.0)), 1),
            "tiros": round(float(xstyle.get("tiros", 11.0)), 1),
            "corners": round(float(xstyle.get("corners", 4.5)), 1),
            "posesion": round(float(xstyle.get("posesion") or 50.0), 1),
            "amarillas": round(float(xstyle.get("amarillas", 2.0)), 2),
            "rojas": round(float(xstyle.get("rojas", 0.1)), 3),
        },
        "forma_reciente": {
            "partidos": forma_kp.get("partidos_analizados", 0),
            "faltas_media": round(float(forma_kp.get("faltas_media", 12.0)), 1),
            "tarjetas_media": round(float(forma_kp.get("tarjetas_media", 2.0)), 2),
            "momentum": round(float(forma_ctx.get("momentum_score", 0.5)), 3),
        },
        "contexto": {
            "urgencia": round(float(compet.get("urgency_score") or 0.5), 3),
            "fatiga": round(float(compet.get("fatigue_score") or 0.2), 3),
            "dias_descanso": calendario.get("days_since_last") or 7,
            "factor_xfaltas": round(float(factors.get("xfouls_factor", 1.0)), 4),
        },
    }


# ---------------------------------------------------------------------------
# Paso 5: Ensamblar contrato
# ---------------------------------------------------------------------------


def _assemble_contract(
    state: dict,
    kp: dict,
    ctx_payload: dict,
    eq_local: str,
    eq_visit: str,
    arbitro: str | None,
    arbitraje_source: str,
    jornada: int,
    fecha_partido: str | None,
    market_input_model: dict | None,
    market_used: list[str],
    odds_scraped_at: str | None,
    warnings: list[str],
) -> dict:
    """Ensambla el dict anidado completo (contrato)."""
    partidos = state["partidos"]
    xstyles = state["xstyles"]

    # GMM del arbitro
    perfiles_gmm = state.get("perfiles_gmm") or calcular_perfiles_gmm(partidos)
    arb_estadisticas = get_perfil_gmm_o_default(arbitro, perfiles_gmm)
    arb_interaccion = ctx_payload.get("arbitraje", {}).get("interaccion_equipos", {})

    # Promedios contextuales del árbitro (clean/heavy) desde ref_perfiles
    refs = state.get("ref_perfiles") or calcular_perfiles_arbitros(partidos)
    ref_data = refs.get(arbitro, {}) if arbitro else {}
    if ref_data.get("fouls_clean_avg") is not None:
        arb_estadisticas["fouls_clean_avg"] = ref_data["fouls_clean_avg"]
    if ref_data.get("fouls_heavy_avg") is not None:
        arb_estadisticas["fouls_heavy_avg"] = ref_data["fouls_heavy_avg"]

    # Metricas del knowledge pack
    em = kp.get("expected_metrics", {})
    xgoals = em.get("xgoals", {})
    xfouls_m = em.get("xfouls", {})
    xposesion = em.get("xposesion", {})
    agresividad = em.get("agresividad_volumen", {})
    xvol = em.get("xvolumen_eventos", {})

    # Agresividad normalizada [0, 1]
    agg_l = float(agresividad.get("local", 12.0))
    agg_v = float(agresividad.get("visitante", 12.0))
    agg_total_norm = max(0.0, min(1.0, (agg_l + agg_v) / _AGG_SCALE_TOTAL))
    is_derby = frozenset({eq_local, eq_visit}) in _DERBIES

    # Extraer urgencia y fatiga del contexto competitivo
    comp_local = (
        ctx_payload.get("competitivo", {}).get("local", {}).get("competitividad", {})
    )
    comp_visit = (
        ctx_payload.get("competitivo", {})
        .get("visitante", {})
        .get("competitividad", {})
    )

    # Labels del partido
    labels = build_labels_category(
        kp,
        is_derby=is_derby,
        aggressiveness_total=agg_total_norm,
        urgency_home=comp_local.get("urgency_score"),
        urgency_away=comp_visit.get("urgency_score"),
        fatigue_home=float(comp_local.get("fatigue_score", 0.2)),
        fatigue_away=float(comp_visit.get("fatigue_score", 0.2)),
    )
    h2h_media, h2h_n = _h2h_faltas(partidos, eq_local, eq_visit)
    temporada_str = ctx_payload.get("temporada", {}).get("season", "2025-26")

    contract = {
        "partido": {
            "equipo_local": eq_local,
            "equipo_visitante": eq_visit,
            "jornada": jornada,
            "temporada": temporada_str,
            "fecha": fecha_partido or datetime.now().strftime("%Y-%m-%d"),
        },
        "arbitro": {
            "nombre": arbitro or "unknown",
            "estadisticas": arb_estadisticas,
            "home_bias": arb_interaccion.get("home_bias", 0.5),
            "interaccion_local": {
                "delta_faltas": arb_interaccion.get("delta_local", 0.0),
                "partidos": arb_interaccion.get("n_partidos_local", 0),
            },
            "interaccion_visitante": {
                "delta_faltas": arb_interaccion.get("delta_visitante", 0.0),
                "partidos": arb_interaccion.get("n_partidos_visitante", 0),
            },
        },
        "equipos": {
            "local": _build_team_block(
                xstyle=xstyles.get(eq_local, {}),
                forma_kp=kp.get("forma", {}).get("local", {}),
                ctx_payload=ctx_payload.get("competitivo", {}).get("local", {}),
            ),
            "visitante": _build_team_block(
                xstyle=xstyles.get(eq_visit, {}),
                forma_kp=kp.get("forma", {}).get("visitante", {}),
                ctx_payload=ctx_payload.get("competitivo", {}).get("visitante", {}),
            ),
        },
        "metricas_esperadas": {
            "xgoles": {
                "local": xgoals.get("xg_local", 0.0),
                "visitante": xgoals.get("xg_visitante", 0.0),
                "total": xgoals.get("xg_total", 0.0),
            },
            "xfaltas": {
                "local": xfouls_m.get("local", 12.5),
                "visitante": xfouls_m.get("visitante", 12.5),
                "total": xfouls_m.get("total", 25.0),
                "media_liga": xfouls_m.get("avg_liga", 25.2),
            },
            "xposesion": {
                "local": xposesion.get("posesion_local", 50.0),
                "visitante": xposesion.get("posesion_visitante", 50.0),
            },
            "agresividad": {
                "local": round(max(0.0, min(1.0, agg_l / _AGG_SCALE_TEAM)), 4),
                "visitante": round(max(0.0, min(1.0, agg_v / _AGG_SCALE_TEAM)), 4),
                "total": round(
                    max(0.0, min(1.0, (agg_l + agg_v) / _AGG_SCALE_TOTAL)), 4
                ),
            },
            "volumen": {
                "tiros_total": xvol.get("shots_total", 22.0),
                "corners_total": xvol.get("corners_total", 9.0),
                "pace_index": round(
                    float(xvol.get("shots_total", 22.0))
                    + float(xvol.get("corners_total", 9.0)),
                    1,
                ),
            },
        },
        "mercado": _build_mercado_block(market_input_model or {}),
        "contexto_partido": {
            "intensidad_esperada": labels.get("intensidad_esperada", "media"),
            "riesgo_disciplinario": labels.get("riesgo_disciplinario", "medio"),
            "is_derby": is_derby,
            "season_phase": round(jornada / 38.0, 4),
            "tipo_partido": labels.get("tipo_partido", "PARTIDO EQUILIBRADO"),
            "sesgo_mercado": labels.get("sesgo_mercado_vs_modelo", "alineado"),
            "h2h_faltas_media": h2h_media,
            "h2h_partidos": h2h_n,
        },
        "_meta": {
            "schema_version": "contract_v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_snapshot": {
                "matches_updated_at": state["updated_at"],
                "odds_scraped_at": odds_scraped_at,
            },
            "input": {
                "equipo_local": eq_local,
                "equipo_visitante": eq_visit,
                "jornada": jornada,
                "arbitro": arbitro,
                "arbitraje_source": arbitraje_source,
            },
            "narrative": kp.get("narrative", []),
            "market_used_for_model": market_used,
            "warnings": warnings,
        },
    }

    contract["_meta"]["quality"] = build_quality_category(contract, warnings)
    return contract


# ---------------------------------------------------------------------------
# Paso 6: Flatten (contrato anidado → flat dict para el modelo)
# ---------------------------------------------------------------------------


def _flatten(raw: dict) -> dict:
    """Convierte el dict anidado a un flat dict de features para el modelo."""
    partido = raw.get("partido", {})
    arbitro = raw.get("arbitro", {})
    equipos = raw.get("equipos", {})
    metricas = raw.get("metricas_esperadas", {})
    mercado = raw.get("mercado", {})
    contexto_partido = raw.get("contexto_partido", {})

    local = equipos.get("local", {})
    visitante = equipos.get("visitante", {})
    local_temp = local.get("temporada_completa", {})
    visitante_temp = visitante.get("temporada_completa", {})
    local_forma = local.get("forma_reciente", {})
    visitante_forma = visitante.get("forma_reciente", {})
    local_ctx = local.get("contexto", {})
    visitante_ctx = visitante.get("contexto", {})

    xgoles = metricas.get("xgoles", {})
    xfaltas = metricas.get("xfaltas", {})
    xposesion = metricas.get("xposesion", {})
    agresividad = metricas.get("agresividad", {})
    volumen = metricas.get("volumen", {})
    pace_index_val = float(volumen.get("pace_index", 31.0))

    resultado = mercado.get("resultado", {})
    goles_ou = mercado.get("goles_ou", {})
    faltas_ou = mercado.get("faltas_total_ou", {})
    derivadas = mercado.get("derivadas", {})

    arb_stats = arbitro.get("estadisticas", {})
    arb_int_local = arbitro.get("interaccion_local", {})
    arb_int_visitante = arbitro.get("interaccion_visitante", {})

    home_rank = int(local.get("clasificacion", {}).get("posicion") or 10)
    away_rank = int(visitante.get("clasificacion", {}).get("posicion") or 10)

    p_h = float(resultado.get("prob_local", 0.40))
    p_d = float(resultado.get("prob_empate", 0.25))
    p_a = float(resultado.get("prob_visitante", 0.35))

    def _poss_norm(v: float) -> float:
        return v / 100.0 if v > 1.0 else v

    return {
        "home_team": str(partido.get("equipo_local", "unknown")),
        "away_team": str(partido.get("equipo_visitante", "unknown")),
        "referee": str(arbitro.get("nombre", "unknown")),
        "matchday": int(partido.get("jornada", 19)),
        "season": str(partido.get("temporada", "2025-26")),
        "date": str(partido.get("fecha", "")),
        "home_fouls_committed_avg": float(local_temp.get("faltas_cometidas", 12.0)),
        "home_fouls_suffered_avg": float(local_temp.get("faltas_provocadas", 12.0)),
        "away_fouls_committed_avg": float(visitante_temp.get("faltas_cometidas", 12.0)),
        "away_fouls_suffered_avg": float(visitante_temp.get("faltas_provocadas", 12.0)),
        "home_fouls_committed_curr": float(local_forma.get("faltas_media", 12.0)),
        "away_fouls_committed_curr": float(visitante_forma.get("faltas_media", 12.0)),
        "home_shots_curr": float(local_temp.get("tiros", pace_index_val * 0.4)),
        "away_shots_curr": float(visitante_temp.get("tiros", pace_index_val * 0.4)),
        "home_corners_curr": float(local_temp.get("corners", pace_index_val * 0.15)),
        "away_corners_curr": float(
            visitante_temp.get("corners", pace_index_val * 0.15)
        ),
        "home_yellows_avg": float(local_temp.get("amarillas", 2.0)),
        "away_yellows_avg": float(visitante_temp.get("amarillas", 2.0)),
        "home_reds_avg": float(local_temp.get("rojas", 0.1)),
        "away_reds_avg": float(visitante_temp.get("rojas", 0.1)),
        "home_rank_hist": float(home_rank),
        "away_rank_hist": float(away_rank),
        "home_rank_curr": home_rank,
        "away_rank_curr": away_rank,
        "rank_diff_norm": round(abs(home_rank - away_rank) / 19.0, 4),
        "season_phase": float(contexto_partido.get("season_phase", 0.5)),
        "is_derby": bool(contexto_partido.get("is_derby", False)),
        "pace_index_curr": pace_index_val,
        "home_possession": _poss_norm(float(xposesion.get("local", 50.0))),
        "away_possession": _poss_norm(float(xposesion.get("visitante", 50.0))),
        "home_xg": float(xgoles.get("local", 0.0)),
        "away_xg": float(xgoles.get("visitante", 0.0)),
        "xg_diff": float(xgoles.get("local", 0.0))
        - float(xgoles.get("visitante", 0.0)),
        "xfouls_home": float(xfaltas.get("local", 12.5)),
        "xfouls_away": float(xfaltas.get("visitante", 12.5)),
        "aggressiveness_volume_home": float(agresividad.get("local", 0.5)),
        "aggressiveness_volume_away": float(agresividad.get("visitante", 0.5)),
        "aggressiveness_norm_total": float(agresividad.get("total", 0.5)),
        "fouls_provoked_home": float(local_temp.get("faltas_provocadas", 12.0)),
        "fouls_provoked_away": float(visitante_temp.get("faltas_provocadas", 12.0)),
        "forma_fouls_home": float(local_forma.get("faltas_media", 12.0)),
        "forma_fouls_away": float(visitante_forma.get("faltas_media", 12.0)),
        "urgency_home": float(local_ctx.get("urgencia", 0.5)),
        "urgency_away": float(visitante_ctx.get("urgencia", 0.5)),
        "momentum_home": float(local_forma.get("momentum", 0.5)),
        "momentum_away": float(visitante_forma.get("momentum", 0.5)),
        "fatigue_home": float(local_ctx.get("fatiga", 0.2)),
        "fatigue_away": float(visitante_ctx.get("fatiga", 0.2)),
        "days_rest_home": float(local_ctx.get("dias_descanso") or 7),
        "days_rest_away": float(visitante_ctx.get("dias_descanso") or 7),
        "xfouls_factor_home": float(local_ctx.get("factor_xfaltas", 1.0)),
        "xfouls_factor_away": float(visitante_ctx.get("factor_xfaltas", 1.0)),
        "referee_mu_permisivo": float(arb_stats.get("mu_permisivo", 22.0)),
        "referee_mu_estricto": float(arb_stats.get("mu_estricto", 30.0)),
        "referee_sigma_permisivo": float(arb_stats.get("sigma_permisivo", 4.0)),
        "referee_sigma_estricto": float(arb_stats.get("sigma_estricto", 4.0)),
        "referee_peso_estricto": float(arb_stats.get("peso_estricto", 0.5)),
        "referee_n_partidos": int(arb_stats.get("partidos_arbitrados", 0)),
        # D1: propagate shrinkage flag so ensemble._register_profiles_from_features
        # can read it directly without re-inferring from n_partidos (REQ-1).
        "referee_is_shrunk": bool(arb_stats.get("is_shrunk", False)),
        "ref_home_delta": float(arb_int_local.get("delta_faltas", 0.0)),
        "ref_away_delta": float(arb_int_visitante.get("delta_faltas", 0.0)),
        "ref_pair_delta_sum": (
            float(arb_int_local.get("delta_faltas", 0.0))
            + float(arb_int_visitante.get("delta_faltas", 0.0))
        ),
        "ref_pair_samples": float(arb_int_local.get("partidos", 0)),
        "has_market_odds": bool(derivadas.get("has_market_odds", False)),
        "market_home_win_prob": p_h,
        "market_draw_prob": p_d,
        "market_away_win_prob": p_a,
        "market_favorite_prob": float(
            derivadas.get("market_favorite_prob", max(p_h, p_a))
        ),
        "market_balance": float(derivadas.get("market_balance", 1.0 - abs(p_h - p_a))),
        "market_entropy": float(derivadas.get("market_entropy", 1.0)),
        "market_ou25_over_prob": float(goles_ou.get("prob_over", 0.50)),
        "market_ou25_under_prob": float(goles_ou.get("prob_under", 0.50)),
        "foul_market_prob_over": float(faltas_ou.get("prob_over", 0.50)),
        "foul_market_implied_mean": float(faltas_ou.get("linea", 24.5)),
        # Promedio contextual del árbitro con equipos limpios (< 11.5 faltas/partido)
        # Usado como floor en ensemble._enrich_match() — no es feature del modelo.
        "referee_clean_avg": float(
            arb_stats.get(
                "fouls_clean_avg",
                arb_stats.get("mu_permisivo", 22.0)
                * (1.0 - float(arb_stats.get("peso_estricto", 0.5)))
                + arb_stats.get("mu_estricto", 30.0)
                * float(arb_stats.get("peso_estricto", 0.5)),
            )
        ),
        # Árbitro × equipo — nuevas features
        "referee_avg_fouls": round(
            float(arb_stats.get("mu_permisivo", 22.0))
            * (1.0 - float(arb_stats.get("peso_estricto", 0.5)))
            + float(arb_stats.get("mu_estricto", 30.0))
            * float(arb_stats.get("peso_estricto", 0.5)),
            2,
        ),
        "referee_home_bias": float(arbitro.get("home_bias", 0.5)),
        "referee_team_committed_home": round(
            float(local_temp.get("faltas_cometidas", 12.0))
            + float(arb_int_local.get("delta_faltas", 0.0)),
            2,
        ),
        "referee_team_committed_away": round(
            float(visitante_temp.get("faltas_cometidas", 12.0))
            + float(arb_int_visitante.get("delta_faltas", 0.0)),
            2,
        ),
        "intensidad_esperada": str(
            contexto_partido.get("intensidad_esperada", "media")
        ),
        "riesgo_disciplinario": str(
            contexto_partido.get("riesgo_disciplinario", "medio")
        ),
        "h2h_faltas_media": float(contexto_partido.get("h2h_faltas_media") or 25.0),
        "h2h_partidos": int(contexto_partido.get("h2h_partidos", 0)),
    }


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------


def _make_registry(
    features_profile: str | None,
    features_config: str | dict | None,
) -> FeatureRegistry:
    """Construye el FeatureRegistry desde el perfil y config indicados."""
    if isinstance(features_config, dict):
        return FeatureRegistry(config=features_config, profile=features_profile)
    return FeatureRegistry(yaml_path=features_config or None, profile=features_profile)


def _run_pipeline(
    *,
    state: dict[str, Any],
    equipo_local_input: str,
    equipo_visitante_input: str,
    jornada: int | None,
    arbitro_input: str | None,
    arbitraje_source: str,
    cuotas_prepartido: dict[str, Any] | None,
    skip_market_fetch: bool,
    fecha_partido_input: str | None,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo y devuelve el contrato anidado."""
    eq_local, eq_visit, arbitro, jornada_final, warnings = _resolve_inputs(
        state,
        equipo_local_input,
        equipo_visitante_input,
        arbitro_input,
        jornada,
    )
    kp, ctx_payload, xf = _compute_expected_stats(
        state,
        eq_local,
        eq_visit,
        arbitro,
        jornada_final,
        arbitraje_source,
        fecha_partido_input,
    )
    kp, market_input_model, market_used, odds_scraped_at = _fetch_and_apply_market(
        state,
        kp,
        eq_local,
        eq_visit,
        cuotas_prepartido,
        skip_market_fetch,
        warnings,
        fecha_partido=fecha_partido_input,
    )
    return _assemble_contract(
        state,
        kp,
        ctx_payload,
        eq_local,
        eq_visit,
        arbitro,
        arbitraje_source,
        jornada_final,
        fecha_partido_input,
        market_input_model,
        market_used,
        odds_scraped_at,
        warnings,
    )


def build_features(
    *,
    state: dict[str, Any],
    equipo_local_input: str,
    equipo_visitante_input: str,
    jornada: int | None = None,
    arbitro_input: str | None = None,
    arbitraje_source: str = "unknown",
    cuotas_prepartido: dict[str, Any] | None = None,
    skip_market_fetch: bool = False,
    fecha_partido_input: str | None = None,
    features_profile: str | None = None,
    features_config: str | dict | None = None,
) -> dict[str, Any]:
    """Genera el flat dict de features listo para el modelo de prediccion.

    El calendario multi-competicion se obtiene automaticamente de cal_index
    (tabla liga_calendar en Supabase) dentro del state.

    El resultado se filtra via FeatureRegistry segun el perfil activo en features.yaml.
    Features de grupos desactivados se omiten del dict de salida.
    """
    registry = _make_registry(features_profile, features_config)
    contract = _run_pipeline(
        state=state,
        equipo_local_input=equipo_local_input,
        equipo_visitante_input=equipo_visitante_input,
        jornada=jornada,
        arbitro_input=arbitro_input,
        arbitraje_source=arbitraje_source,
        cuotas_prepartido=cuotas_prepartido,
        skip_market_fetch=skip_market_fetch,
        fecha_partido_input=fecha_partido_input,
    )
    flat = _flatten(contract)
    return registry.filter(flat)


def build_detailed(
    *,
    state: dict[str, Any],
    equipo_local_input: str,
    equipo_visitante_input: str,
    jornada: int | None = None,
    arbitro_input: str | None = None,
    arbitraje_source: str = "unknown",
    cuotas_prepartido: dict[str, Any] | None = None,
    skip_market_fetch: bool = False,
    fecha_partido_input: str | None = None,
) -> dict[str, Any]:
    """Genera datos detallados del partido en formato anidado (para reports/debug).

    Retorna el contrato completo con todos los bloques anidados.
    Para el flat dict filtrado del modelo, usar build_features().
    """
    return _run_pipeline(
        state=state,
        equipo_local_input=equipo_local_input,
        equipo_visitante_input=equipo_visitante_input,
        jornada=jornada,
        arbitro_input=arbitro_input,
        arbitraje_source=arbitraje_source,
        cuotas_prepartido=cuotas_prepartido,
        skip_market_fetch=skip_market_fetch,
        fecha_partido_input=fecha_partido_input,
    )
