"""
Genera un PDF exhaustivo con TODOS los datos usados para la predicción:
- Contrato completo (partido, árbitro, equipos, métricas, mercado, contexto)
- Features planas del modelo (input directo al ensemble)
- Datos históricos: partidos del árbitro, forma equipos, H2H
- Predicción: ensemble, PMF, Over/Under, desglose por equipo
"""
from __future__ import annotations

import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "featuresGenerator"))
sys.path.insert(0, str(ROOT / "predictionModels"))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)

CHECKPOINT_DIR = ROOT / "predictionModels" / "checkpoints" / "ensemble"


def _fmt(v, nd=2):
    if v is None:
        return "N/D"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v, nd=1):
    if v is None:
        return "N/D"
    return f"{float(v)*100:.{nd}f}%"


def _total_fouls(match):
    return float(match["home"].get("fouls", 0) + match["away"].get("fouls", 0))


def _team_fouls(match, team, role="committed"):
    if match["home"]["name"] == team:
        return float(match["home" if role == "committed" else "away"].get("fouls", 0))
    if match["away"]["name"] == team:
        return float(match["away" if role == "committed" else "home"].get("fouls", 0))
    return 0.0


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("SectionTitle", parent=styles["Heading2"],
                              textColor=colors.HexColor("#1a3c5e"),
                              spaceBefore=14, spaceAfter=4))
    styles.add(ParagraphStyle("SubSection", parent=styles["Heading3"],
                              textColor=colors.HexColor("#2d6da3"),
                              spaceBefore=10, spaceAfter=2))
    styles.add(ParagraphStyle("CellText", parent=styles["Normal"],
                              fontSize=7.5, leading=9))
    styles.add(ParagraphStyle("SmallItalic", parent=styles["Italic"],
                              fontSize=7, leading=9, textColor=colors.grey))
    return styles


def make_table(title, rows, styles, elems, col_widths=None):
    if col_widths is None:
        col_widths = [6.8 * cm, 10.5 * cm]

    safe_rows = []
    for r in rows:
        safe_rows.append([
            Paragraph(str(c), styles["CellText"]) if not isinstance(c, Paragraph) else c
            for c in r
        ])

    header = [Paragraph("<b>Campo</b>", styles["CellText"]),
              Paragraph("<b>Valor</b>", styles["CellText"])]

    t = Table([header] + safe_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f9fc"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elems.append(Paragraph(f"<b>{title}</b>", styles["SubSection"]))
    elems.append(t)
    elems.append(Spacer(1, 6))


def make_wide_table(title, headers, rows, styles, elems, col_widths=None):
    safe_rows = []
    for r in rows:
        safe_rows.append([Paragraph(str(c), styles["CellText"]) for c in r])
    h = [Paragraph(f"<b>{c}</b>", styles["CellText"]) for c in headers]
    ncols = len(headers)
    if col_widths is None:
        col_widths = [17.3 * cm / ncols] * ncols

    t = Table([h] + safe_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f9fc"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    elems.append(Paragraph(f"<b>{title}</b>", styles["SubSection"]))
    elems.append(t)
    elems.append(Spacer(1, 6))


def generate_pdf(
    equipo_local: str = "Real Madrid",
    equipo_visitante: str = "Ath Madrid",
    jornada: int = 29,
    arbitro_name: str | None = "Munuera Montero",
    fecha_partido: str | None = "2026-03-22",
):
    from core.state_cache import get_state
    from assembly.feature_assembler import build_detailed, _flatten

    state = get_state()
    contract = build_detailed(
        state=state,
        equipo_local_input=equipo_local,
        equipo_visitante_input=equipo_visitante,
        jornada=jornada,
        arbitro_input=arbitro_name,
        fecha_partido_input=fecha_partido,
    )

    partido = contract["partido"]
    arbitro = contract["arbitro"]
    equipos = contract["equipos"]
    metricas = contract["metricas_esperadas"]
    mercado = contract["mercado"]
    contexto = contract["contexto_partido"]
    meta = contract["_meta"]

    home = partido["equipo_local"]
    away = partido["equipo_visitante"]
    referee = arbitro["nombre"]

    feat = _flatten(contract)

    from src.models.ensemble import FoulPredictionEnsemble
    from scripts.train import load_training_parquet, build_team_stats, split_train_gating

    parquet_path = ROOT / "predictionModels" / "data" / "training.parquet"
    cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text(encoding="utf-8"))

    print("  Cargando datos de entrenamiento...")
    train_data = load_training_parquet(parquet_path)
    train_feats, gating_feats = split_train_gating(train_data, "2025-26")
    avg_c, avg_s, avg_r = build_team_stats(train_feats)

    print("  Entrenando ensemble (incluye team_regressor)...")
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.fit(train_feats, avg_c, avg_s, avg_r,
                 fit_team_models=True, gating_matches=gating_feats or None)

    pred = ensemble.predict(feat)
    team_pred = ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True)
    if team_pred is None:
        team_pred = {
            "home_expected": pred.expected_fouls * 0.45,
            "away_expected": pred.expected_fouls * 0.55,
            "total_expected": pred.expected_fouls,
            "raw_total_expected": pred.expected_fouls,
            "raw_home_expected": pred.expected_fouls * 0.45,
            "raw_away_expected": pred.expected_fouls * 0.55,
            "coherent_constraint_error": 0.0,
            "total_pmf": pred.pmf_total,
        }

    from core.state_cache import get_state
    state = get_state()
    partidos = state["partidos"]

    ref_matches = sorted(
        [p for p in partidos if (p.get("referee") or "").strip() == referee],
        key=lambda p: p["date"], reverse=True)
    ref_with_home = [p for p in ref_matches
                     if p["home"]["name"] == home or p["away"]["name"] == home]
    ref_with_away = [p for p in ref_matches
                     if p["home"]["name"] == away or p["away"]["name"] == away]

    h2h = sorted(
        [p for p in partidos
         if frozenset({p["home"]["name"], p["away"]["name"]}) == frozenset({home, away})],
        key=lambda p: p["date"], reverse=True)

    home_last = sorted(
        [p for p in partidos if p["home"]["name"] == home or p["away"]["name"] == home],
        key=lambda p: p["date"], reverse=True)[:5]
    away_last = sorted(
        [p for p in partidos if p["home"]["name"] == away or p["away"]["name"] == away],
        key=lambda p: p["date"], reverse=True)[:5]

    # --- PDF ---
    styles = build_styles()
    out_dir = ROOT / "predictionModels" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"informe_completo_{ts}_{home.lower().replace(' ','_')}_vs_{away.lower().replace(' ','_')}.pdf"

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            rightMargin=1.4*cm, leftMargin=1.4*cm,
                            topMargin=1.2*cm, bottomMargin=1.0*cm)
    elems: list[Any] = []

    # ===== PORTADA =====
    elems.append(Spacer(1, 40))
    elems.append(Paragraph("INFORME COMPLETO DE PREDICCIÓN DE FALTAS", styles["Title"]))
    elems.append(Spacer(1, 10))
    elems.append(Paragraph(
        f"<b>{home} vs {away}</b><br/>"
        f"Jornada {partido['jornada']} · Temporada {partido['temporada']} · {partido['fecha']}<br/>"
        f"Árbitro: <b>{referee}</b>",
        styles["Heading3"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} · Schema: {meta['schema_version']}",
        styles["SmallItalic"]))
    elems.append(Spacer(1, 20))

    # ===== 1. PREDICCION =====
    elems.append(Paragraph("1. RESULTADO DE LA PREDICCIÓN", styles["SectionTitle"]))

    make_table("1.1 Predicción principal del ensemble", [
        ["Faltas totales esperadas (raw)", _fmt(pred.expected_fouls)],
        [f"Faltas {home} (reconciliado)", _fmt(team_pred["home_expected"])],
        [f"Faltas {away} (reconciliado)", _fmt(team_pred["away_expected"])],
        ["Total reconciliado (home + away)", _fmt(team_pred["total_expected"])],
        ["Error de coherencia (total - H - A)", _fmt(team_pred.get("coherent_constraint_error", 0))],
        ["Raw total (antes de reconciliar)", _fmt(team_pred.get("raw_total_expected"))],
        ["Raw home", _fmt(team_pred.get("raw_home_expected"))],
        ["Raw away", _fmt(team_pred.get("raw_away_expected"))],
        ["P(árbitro modo estricto)", _pct(pred.referee_strict_prob)],
        ["Peso Naive Bayes", _fmt(pred.weights[0], 4)],
        ["Peso Regresión", _fmt(pred.weights[1], 4)],
        ["Peso ANFIS", _fmt(pred.weights[2], 4)],
    ], styles, elems)

    ou_lines = [19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 31.5]
    ou_table = team_pred["total_pmf"].over_under_table(ou_lines) if team_pred.get("total_pmf") else pred.over_under
    ou_rows = []
    for line in ou_lines:
        if line in ou_table:
            p_o, p_u = ou_table[line]
            tag = " ← línea mercado" if abs(line - feat.get("foul_market_implied_mean", 24.5)) < 1.0 else ""
            ou_rows.append([f"OU {line:.1f}", _pct(p_o), _pct(p_u), tag])
    make_wide_table("1.2 Over/Under completo",
                    ["Línea", "Over", "Under", "Nota"],
                    ou_rows, styles, elems,
                    col_widths=[3*cm, 4*cm, 4*cm, 6.3*cm])

    elems.append(PageBreak())

    # ===== 2. CONTRATO: PARTIDO =====
    elems.append(Paragraph("2. DATOS DEL CONTRATO (INPUT AL MODELO)", styles["SectionTitle"]))

    make_table("2.1 Identificación del partido", [
        ["Equipo local", home],
        ["Equipo visitante", away],
        ["Jornada", str(partido["jornada"])],
        ["Temporada", partido["temporada"]],
        ["Fecha", partido["fecha"]],
        ["Es derby", _fmt(contexto["is_derby"])],
        ["Season phase", _fmt(contexto["season_phase"], 4)],
        ["Tipo de partido", contexto.get("tipo_partido", "")],
        ["Intensidad esperada", contexto["intensidad_esperada"]],
        ["Riesgo disciplinario", contexto["riesgo_disciplinario"]],
        ["Sesgo mercado vs modelo", contexto.get("sesgo_mercado", "")],
        ["H2H faltas media", _fmt(contexto["h2h_faltas_media"])],
        ["H2H partidos previos", str(contexto["h2h_partidos"])],
    ], styles, elems)

    # ===== 2.2 ARBITRO =====
    arb_stats = arbitro["estadisticas"]
    arb_int_l = arbitro["interaccion_local"]
    arb_int_v = arbitro["interaccion_visitante"]
    make_table("2.2 Perfil del árbitro (GMM bimodal)", [
        ["Nombre", referee],
        ["Partidos arbitrados", str(arb_stats["partidos_arbitrados"])],
        ["μ permisivo (media faltas modo permisivo)", _fmt(arb_stats["mu_permisivo"])],
        ["σ permisivo (desv. estándar modo permisivo)", _fmt(arb_stats["sigma_permisivo"])],
        ["μ estricto (media faltas modo estricto)", _fmt(arb_stats["mu_estricto"])],
        ["σ estricto (desv. estándar modo estricto)", _fmt(arb_stats["sigma_estricto"])],
        ["Peso estricto (prob. a priori de estricto)", _fmt(arb_stats["peso_estricto"], 4)],
        ["Perfil con shrinkage bayesiano", _fmt(arb_stats.get("is_shrunk"))],
        [f"Delta faltas con {home}", f"{arb_int_l['delta_faltas']:+.3f} ({arb_int_l['partidos']} partidos)"],
        [f"Delta faltas con {away}", f"{arb_int_v['delta_faltas']:+.3f} ({arb_int_v['partidos']} partidos)"],
    ], styles, elems)

    # ===== 2.3 EQUIPOS =====
    for side, label in [("local", home), ("visitante", away)]:
        eq = equipos[side]
        cl = eq["clasificacion"]
        tc = eq["temporada_completa"]
        fr = eq["forma_reciente"]
        cx = eq["contexto"]
        make_table(f"2.3 Equipo {'local' if side == 'local' else 'visitante'}: {label}", [
            ["Posición en la clasificación", str(cl["posicion"])],
            ["Puntos", str(cl["puntos"])],
            ["Partidos jugados", str(cl["partidos_jugados"])],
            ["Diferencia de goles", str(cl["diferencia_goles"])],
            ["", ""],
            ["--- Temporada completa (medias) ---", ""],
            ["Faltas cometidas / partido", _fmt(tc["faltas_cometidas"], 1)],
            ["Faltas provocadas / partido", _fmt(tc["faltas_provocadas"], 1)],
            ["Tiros / partido", _fmt(tc["tiros"], 1)],
            ["Córners / partido", _fmt(tc["corners"], 1)],
            ["Posesión media", f"{tc['posesion']}%"],
            ["Amarillas / partido", _fmt(tc["amarillas"])],
            ["Rojas / partido", _fmt(tc["rojas"], 3)],
            ["", ""],
            ["--- Forma reciente (últimos partidos) ---", ""],
            ["Partidos analizados", str(fr["partidos"])],
            ["Faltas media reciente", _fmt(fr["faltas_media"], 1)],
            ["Tarjetas media reciente", _fmt(fr["tarjetas_media"])],
            ["Momentum", _fmt(fr["momentum"], 3)],
            ["", ""],
            ["--- Contexto competitivo ---", ""],
            ["Urgencia", _fmt(cx["urgencia"], 3)],
            ["Fatiga", _fmt(cx["fatiga"], 3)],
            ["Días de descanso", str(cx["dias_descanso"])],
            ["Factor xFaltas", _fmt(cx["factor_xfaltas"], 4)],
        ], styles, elems)

    elems.append(PageBreak())

    # ===== 2.4 METRICAS ESPERADAS =====
    xg = metricas["xgoles"]
    xf = metricas["xfaltas"]
    xp = metricas["xposesion"]
    ag = metricas["agresividad"]
    vol = metricas["volumen"]
    make_table("2.4 Métricas esperadas (calculadas por featuresGenerator)", [
        ["xGoles local", _fmt(xg["local"])],
        ["xGoles visitante", _fmt(xg["visitante"])],
        ["xGoles total", _fmt(xg["total"])],
        ["", ""],
        ["xFaltas local", _fmt(xf["local"], 1)],
        ["xFaltas visitante", _fmt(xf["visitante"], 1)],
        ["xFaltas total", _fmt(xf["total"], 1)],
        ["xFaltas media liga", _fmt(xf["media_liga"], 1)],
        ["", ""],
        ["xPosesión local", f"{xp['local']}%"],
        ["xPosesión visitante", f"{xp['visitante']}%"],
        ["", ""],
        ["Agresividad normalizada local", _fmt(ag["local"], 4)],
        ["Agresividad normalizada visitante", _fmt(ag["visitante"], 4)],
        ["Agresividad normalizada total", _fmt(ag["total"], 4)],
        ["", ""],
        ["Tiros totales esperados", _fmt(vol["tiros_total"], 1)],
        ["Córners totales esperados", _fmt(vol["corners_total"], 1)],
        ["Pace index (tiros + córners)", _fmt(vol["pace_index"], 1)],
    ], styles, elems)

    # ===== 2.5 MERCADO =====
    res = mercado["resultado"]
    gou = mercado["goles_ou"]
    fou = mercado["faltas_total_ou"]
    der = mercado["derivadas"]
    make_table("2.5 Señales de mercado", [
        ["P(victoria local) sin vig", _pct(res["prob_local"])],
        ["P(empate) sin vig", _pct(res["prob_empate"])],
        ["P(victoria visitante) sin vig", _pct(res["prob_visitante"])],
        ["Entropía 1X2", _fmt(res["entropia"], 4)],
        ["", ""],
        ["Línea goles O/U", _fmt(gou["linea"], 1)],
        ["P(over goles)", _pct(gou["prob_over"])],
        ["P(under goles)", _pct(gou["prob_under"])],
        ["", ""],
        ["Línea faltas O/U", _fmt(fou["linea"], 1)],
        ["P(over faltas)", _pct(fou["prob_over"])],
        ["P(under faltas)", _pct(fou["prob_under"])],
        ["", ""],
        ["Market entropy (derivada)", _fmt(der["market_entropy"], 4)],
        ["Market balance", _fmt(der["market_balance"], 4)],
        ["Market favorite prob", _pct(der["market_favorite_prob"])],
        ["Has market odds (cuotas reales)", _fmt(der["has_market_odds"])],
    ], styles, elems)

    elems.append(PageBreak())

    # ===== 3. FEATURES PLANAS (INPUT DIRECTO AL ENSEMBLE) =====
    elems.append(Paragraph("3. FEATURES PLANAS (DICT COMPLETO → MODELOS)", styles["SectionTitle"]))
    elems.append(Paragraph(
        "Estas son las features exactas que reciben Naive Bayes, Regresion y ANFIS. "
        "Se generan mediante <i>build.feature_assembler._flatten(contract)</i>.",
        styles["Normal"]))
    elems.append(Spacer(1, 6))

    categories = {
        "Identidad": ["home_team", "away_team", "referee", "matchday", "season", "date"],
        "Temporada completa": [
            "home_fouls_committed_avg", "home_fouls_suffered_avg",
            "away_fouls_committed_avg", "away_fouls_suffered_avg",
            "home_fouls_committed_curr", "away_fouls_committed_curr",
            "home_shots_curr", "away_shots_curr",
            "home_corners_curr", "away_corners_curr",
            "home_yellows_avg", "away_yellows_avg",
            "home_reds_avg", "away_reds_avg",
        ],
        "Clasificación": [
            "home_rank_hist", "away_rank_hist",
            "home_rank_curr", "away_rank_curr",
            "rank_diff_norm",
        ],
        "Contexto de partido": [
            "season_phase", "is_derby", "pace_index_curr",
            "intensidad_esperada", "riesgo_disciplinario",
            "h2h_faltas_media", "h2h_partidos",
        ],
        "Métricas esperadas": [
            "home_possession", "away_possession",
            "home_xg", "away_xg", "xg_diff",
            "xfouls_home", "xfouls_away",
            "aggressiveness_volume_home", "aggressiveness_volume_away",
            "aggressiveness_norm_total",
            "fouls_provoked_home", "fouls_provoked_away",
        ],
        "Forma reciente": [
            "forma_fouls_home", "forma_fouls_away",
        ],
        "Contexto competitivo": [
            "urgency_home", "urgency_away",
            "momentum_home", "momentum_away",
            "fatigue_home", "fatigue_away",
            "days_rest_home", "days_rest_away",
            "xfouls_factor_home", "xfouls_factor_away",
        ],
        "Árbitro GMM": [
            "referee_mu_permisivo", "referee_mu_estricto",
            "referee_sigma_permisivo", "referee_sigma_estricto",
            "referee_peso_estricto", "referee_n_partidos",
        ],
        "Deltas árbitro-equipo": [
            "ref_home_delta", "ref_away_delta",
            "ref_pair_delta_sum", "ref_pair_samples",
        ],
        "Mercado 1X2": [
            "has_market_odds",
            "market_home_win_prob", "market_draw_prob", "market_away_win_prob",
            "market_favorite_prob", "market_balance", "market_entropy",
        ],
        "Mercado goles O/U": [
            "market_ou25_over_prob", "market_ou25_under_prob",
        ],
        "Mercado faltas O/U": [
            "foul_market_prob_over", "foul_market_implied_mean",
            "foul_market_local_prob_over", "foul_market_local_implied_mean",
            "foul_market_vis_prob_over", "foul_market_vis_implied_mean",
        ],
    }

    for cat_name, keys in categories.items():
        rows = []
        for k in keys:
            v = feat.get(k, "—")
            if isinstance(v, float):
                rows.append([k, _fmt(v, 4)])
            elif isinstance(v, bool):
                rows.append([k, _fmt(v)])
            else:
                rows.append([k, str(v)])
        make_table(f"3.x {cat_name}", rows, styles, elems)

    elems.append(PageBreak())

    # ===== 4. DATOS HISTORICOS =====
    elems.append(Paragraph("4. DATOS HISTÓRICOS USADOS", styles["SectionTitle"]))

    # 4.1 Ultimos partidos del arbitro
    ref_recent = ref_matches[:10]
    if ref_recent:
        rows = []
        for p in ref_recent:
            h_name = p["home"]["name"]
            a_name = p["away"]["name"]
            f_h = p["home"].get("fouls", 0)
            f_a = p["away"].get("fouls", 0)
            y_h = p["home"].get("yellow_cards", 0)
            y_a = p["away"].get("yellow_cards", 0)
            rows.append([
                p["date"], f"{h_name} vs {a_name}",
                str(f_h), str(f_a), str(f_h + f_a),
                str(y_h), str(y_a), str(y_h + y_a),
            ])
        make_wide_table(
            f"4.1 Últimos 10 partidos de {referee}",
            ["Fecha", "Partido", "F.Local", "F.Visit", "F.Total", "A.Local", "A.Visit", "A.Total"],
            rows, styles, elems,
            col_widths=[2*cm, 4.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm])

        ref_totals = [_total_fouls(p) for p in ref_matches]
        ref_recent5 = [_total_fouls(p) for p in ref_matches[:5]]
        make_table(f"4.1b Estadísticas de {referee}", [
            ["Total partidos en BD", str(len(ref_matches))],
            ["Media faltas global", _fmt(_safe_mean(ref_totals))],
            ["Media faltas últimos 5", _fmt(_safe_mean(ref_recent5))],
            ["Mín faltas", _fmt(min(ref_totals) if ref_totals else None)],
            ["Máx faltas", _fmt(max(ref_totals) if ref_totals else None)],
            [f"Partidos con {home}", str(len(ref_with_home))],
            [f"Media faltas con {home}", _fmt(_safe_mean([_total_fouls(p) for p in ref_with_home]))],
            [f"Partidos con {away}", str(len(ref_with_away))],
            [f"Media faltas con {away}", _fmt(_safe_mean([_total_fouls(p) for p in ref_with_away]))],
        ], styles, elems)

    # 4.2 Forma reciente del local
    if home_last:
        rows = []
        for p in home_last:
            h_name = p["home"]["name"]
            a_name = p["away"]["name"]
            is_home = h_name == home
            fc = _team_fouls(p, home, "committed")
            fs = _team_fouls(p, home, "suffered")
            g_own = p["home" if is_home else "away"].get("goals", 0)
            g_opp = p["away" if is_home else "home"].get("goals", 0)
            result = "V" if g_own > g_opp else ("E" if g_own == g_opp else "D")
            rows.append([p["date"], f"{h_name} vs {a_name}",
                         "Local" if is_home else "Visit",
                         f"{g_own}-{g_opp}", result,
                         _fmt(fc, 0), _fmt(fs, 0)])
        make_wide_table(
            f"4.2 Últimos 5 partidos de {home}",
            ["Fecha", "Partido", "Condición", "Resultado", "V/E/D", "F.Cometidas", "F.Recibidas"],
            rows, styles, elems,
            col_widths=[2*cm, 4*cm, 2*cm, 2*cm, 1.3*cm, 2.5*cm, 2.5*cm])

    # 4.3 Forma reciente del visitante
    if away_last:
        rows = []
        for p in away_last:
            h_name = p["home"]["name"]
            a_name = p["away"]["name"]
            is_home = h_name == away
            fc = _team_fouls(p, away, "committed")
            fs = _team_fouls(p, away, "suffered")
            g_own = p["home" if is_home else "away"].get("goals", 0)
            g_opp = p["away" if is_home else "home"].get("goals", 0)
            result = "V" if g_own > g_opp else ("E" if g_own == g_opp else "D")
            rows.append([p["date"], f"{h_name} vs {a_name}",
                         "Local" if is_home else "Visit",
                         f"{g_own}-{g_opp}", result,
                         _fmt(fc, 0), _fmt(fs, 0)])
        make_wide_table(
            f"4.3 Últimos 5 partidos de {away}",
            ["Fecha", "Partido", "Condición", "Resultado", "V/E/D", "F.Cometidas", "F.Recibidas"],
            rows, styles, elems,
            col_widths=[2*cm, 4*cm, 2*cm, 2*cm, 1.3*cm, 2.5*cm, 2.5*cm])

    # 4.4 H2H
    if h2h:
        rows = []
        for p in h2h[:10]:
            h_name = p["home"]["name"]
            a_name = p["away"]["name"]
            f_h = p["home"].get("fouls", 0)
            f_a = p["away"].get("fouls", 0)
            g_h = p["home"].get("goals", 0)
            g_a = p["away"].get("goals", 0)
            ref_p = (p.get("referee") or "—")
            rows.append([p["date"], f"{h_name} {g_h}-{g_a} {a_name}",
                         str(f_h), str(f_a), str(f_h + f_a), ref_p])
        make_wide_table(
            f"4.4 H2H: {home} vs {away} (últimos {len(h2h[:10])})",
            ["Fecha", "Resultado", "F.Local", "F.Visit", "F.Total", "Árbitro"],
            rows, styles, elems,
            col_widths=[2*cm, 4.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 4.6*cm])

    elems.append(PageBreak())

    # ===== 5. NARRATIVA Y META =====
    elems.append(Paragraph("5. NARRATIVA DEL MODELO Y METADATOS", styles["SectionTitle"]))

    for i, line in enumerate(meta.get("narrative", []), 1):
        elems.append(Paragraph(f"{i}. {line}", styles["Normal"]))
    elems.append(Spacer(1, 8))

    make_table("5.1 Metadatos de generación", [
        ["Schema version", meta["schema_version"]],
        ["Generado en", meta["generated_at"]],
        ["Datos actualizados en", meta["source_snapshot"]["matches_updated_at"]],
        ["Cuotas scrapeadas en", str(meta["source_snapshot"]["odds_scraped_at"])],
        ["Fuente del árbitro", meta["input"]["arbitraje_source"]],
        ["Coverage score", _fmt(meta.get("quality", {}).get("coverage_score"))],
    ], styles, elems)

    if meta.get("warnings"):
        make_table("5.2 Warnings", [
            [f"Warning {i+1}", w] for i, w in enumerate(meta["warnings"])
        ], styles, elems)

    if meta.get("market_catalog"):
        make_table("5.3 Catálogo de mercados disponibles", [
            [str(i+1), str(m)] for i, m in enumerate(meta["market_catalog"])
        ], styles, elems)

    # ===== 6. CONFIG DEL ENSEMBLE =====
    elems.append(Paragraph("6. CONFIGURACIÓN DEL ENSEMBLE", styles["SectionTitle"]))
    cfg_rows = []
    for section, params in cfg.items():
        if isinstance(params, dict):
            for k, v in params.items():
                cfg_rows.append([f"{section}.{k}", str(v)])
        else:
            cfg_rows.append([section, str(params)])
    make_table("6.1 Hiperparámetros del modelo", cfg_rows, styles, elems)

    # Nota final
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(
        "<i>Este informe contiene la totalidad de datos, features y parámetros utilizados "
        "para generar la predicción. Las estimaciones son probabilísticas y no deterministas. "
        "El valor central puede diferir del resultado real por ruido arbitral, "
        "contexto táctico no observable y eventos de juego impredecibles.</i>",
        styles["SmallItalic"]))

    doc.build(elems)
    return out_path


if __name__ == "__main__":
    path = generate_pdf()
    print(f"\n{'='*65}")
    print(f"  PDF generado: {path}")
    print(f"{'='*65}")
