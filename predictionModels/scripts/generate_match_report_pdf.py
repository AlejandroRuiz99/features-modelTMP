from __future__ import annotations

import json
import statistics
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "featuresGenerator"))
sys.path.insert(0, str(ROOT / "predictionModels"))

from generate import generate_features  # type: ignore
from core.state_cache import get_state  # type: ignore
from scripts.train import build_team_stats, load_training_parquet  # type: ignore
from src.models.ensemble import FoulPredictionEnsemble  # type: ignore


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None:
        return "N/D"
    return f"{v:.{nd}f}"


def _total_fouls(match: dict[str, Any]) -> float:
    return float(match["home"].get("fouls", 0) + match["away"].get("fouls", 0))


def _team_committed(match: dict[str, Any], team: str) -> float:
    if match["home"]["name"] == team:
        return float(match["home"].get("fouls", 0))
    if match["away"]["name"] == team:
        return float(match["away"].get("fouls", 0))
    return 0.0


def _team_suffered(match: dict[str, Any], team: str) -> float:
    if match["home"]["name"] == team:
        return float(match["away"].get("fouls", 0))
    if match["away"]["name"] == team:
        return float(match["home"].get("fouls", 0))
    return 0.0


def _last_n_team_matches(partidos: list[dict[str, Any]], team: str, n: int = 5) -> list[dict[str, Any]]:
    rows = [p for p in partidos if p["home"]["name"] == team or p["away"]["name"] == team]
    rows.sort(key=lambda p: p["date"], reverse=True)
    return rows[:n]


def _h2h_matches(partidos: list[dict[str, Any]], home_team: str, away_team: str, limit: int = 10) -> list[dict[str, Any]]:
    out = []
    for p in partidos:
        h = p["home"]["name"]
        a = p["away"]["name"]
        if (h == home_team and a == away_team) or (h == away_team and a == home_team):
            out.append(p)
    out.sort(key=lambda p: p["date"], reverse=True)
    return out[:limit]


def _recent_referee_matches(partidos: list[dict[str, Any]], referee: str, n: int = 8) -> list[dict[str, Any]]:
    rows = [p for p in partidos if (p.get("referee") or "").strip() == referee]
    rows.sort(key=lambda p: p["date"], reverse=True)
    return rows[:n]


def _trend_label(short_avg: float | None, long_avg: float | None) -> str:
    if short_avg is None or long_avg is None:
        return "Sin datos suficientes"
    delta = short_avg - long_avg
    if delta > 1.0:
        return "Tendencia reciente mas estricta (pita mas faltas)"
    if delta < -1.0:
        return "Tendencia reciente mas permisiva (pita menos faltas)"
    return "Tendencia estable respecto a su media"


def _market_readable(feat: dict[str, Any]) -> list[list[str]]:
    has_odds = bool(feat.get("has_market_odds", False))
    if not has_odds:
        return [
            ["Disponibilidad cuotas", "No hay cuotas reales para este partido"],
            ["Interpretacion", "Las senales de mercado se usan en modo neutral, sin sesgo direccional"],
        ]

    p_h = float(feat.get("market_home_win_prob", 1 / 3))
    p_d = float(feat.get("market_draw_prob", 1 / 3))
    p_a = float(feat.get("market_away_win_prob", 1 / 3))
    p_over = float(feat.get("market_ou25_over_prob", 0.5))
    p_under = float(feat.get("market_ou25_under_prob", 0.5))
    entropy = float(feat.get("market_entropy", 1.0))
    fav = max(p_h, p_a)

    return [
        ["Disponibilidad cuotas", "Si, hay cuotas reales cargadas"],
        ["1X2 Local/Empate/Visitante", f"{p_h:.1%} / {p_d:.1%} / {p_a:.1%}"],
        ["Favoritismo del mercado", f"{fav:.1%}"],
        ["Over/Under 2.5 goles", f"Over {p_over:.1%} / Under {p_under:.1%}"],
        ["Entropia de mercado", f"{entropy:.3f} (alto=partido equilibrado, bajo=favorito claro)"],
    ]


def build_report(home_team: str, away_team: str, matchday: int | None = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    feat = generate_features(
        equipo_local=home_team,
        equipo_visitante=away_team,
        jornada=matchday,
    )
    state = get_state(refresh=False)
    partidos = state["partidos"]

    # Modelo: entrenar en memoria para incluir desglose por equipo
    train_rows = load_training_parquet(ROOT / "predictionModels" / "data" / "training.parquet")
    split_idx = int(len(train_rows) * 0.80)
    train = train_rows[:split_idx]
    gating = train_rows[split_idx:]
    avg_c, avg_s, avg_r = build_team_stats(train)

    cfg = json.loads((ROOT / "predictionModels" / "checkpoints" / "ensemble" / "config.json").read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.fit(train, avg_c, avg_s, avg_r, fit_team_models=True, gating_matches=gating)

    pred = ensemble.predict(feat)
    team_pred = ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True)

    referee = str(feat.get("referee", "Desconocido"))

    # Perfil arbitro
    ref_all = [p for p in partidos if (p.get("referee") or "").strip() == referee]
    ref_all_totals = [_total_fouls(p) for p in ref_all]
    ref_all_mean = _safe_mean(ref_all_totals)

    ref_home = [p for p in ref_all if p["home"]["name"] == home_team or p["away"]["name"] == home_team]
    ref_away = [p for p in ref_all if p["home"]["name"] == away_team or p["away"]["name"] == away_team]
    ref_home_mean = _safe_mean([_total_fouls(p) for p in ref_home])
    ref_away_mean = _safe_mean([_total_fouls(p) for p in ref_away])

    ref_recent = _recent_referee_matches(partidos, referee, n=8)
    ref_recent_totals = [_total_fouls(p) for p in ref_recent]
    ref_recent_5 = list(deque(ref_recent_totals, maxlen=5))
    ref_recent5_mean = _safe_mean(ref_recent_5)
    trend = _trend_label(ref_recent5_mean, ref_all_mean)

    # Rachas equipos
    home_last5 = _last_n_team_matches(partidos, home_team, n=5)
    away_last5 = _last_n_team_matches(partidos, away_team, n=5)

    home_committed = _safe_mean([_team_committed(m, home_team) for m in home_last5])
    home_suffered = _safe_mean([_team_suffered(m, home_team) for m in home_last5])
    away_committed = _safe_mean([_team_committed(m, away_team) for m in away_last5])
    away_suffered = _safe_mean([_team_suffered(m, away_team) for m in away_last5])

    # H2H
    h2h = _h2h_matches(partidos, home_team, away_team, limit=10)
    h2h_totals = [_total_fouls(p) for p in h2h]
    h2h_mean = _safe_mean(h2h_totals)

    # Tipo de partido / contexto
    intensidad = str(feat.get("intensidad_esperada", "media"))
    riesgo = str(feat.get("riesgo_disciplinario", "medio"))
    derby = bool(feat.get("is_derby", False))
    poss_home = float(feat.get("home_possession", 0.5)) * 100.0
    poss_away = float(feat.get("away_possession", 0.5)) * 100.0
    xf_home = float(feat.get("xfouls_home", 12.5))
    xf_away = float(feat.get("xfouls_away", 12.5))

    # PDF
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / "predictionModels" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"informe_prediccion_{today}_{home_team.lower()}_{away_team.lower()}.pdf"

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.2 * cm)
    styles = getSampleStyleSheet()
    elems: list[Any] = []

    elems.append(Paragraph(f"<b>Informe explicativo de prediccion de faltas</b>", styles["Title"]))
    elems.append(Paragraph(f"Partido: <b>{home_team} vs {away_team}</b> | Arbitro previsto: <b>{referee}</b>", styles["Heading3"]))
    elems.append(Paragraph(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elems.append(Spacer(1, 10))

    resumen_txt = (
        f"El modelo estima <b>{pred.expected_fouls:.2f}</b> faltas totales. "
        f"Con reconciliacion por equipos, la distribucion queda en "
        f"<b>{team_pred['home_expected']:.2f}</b> para {home_team} y "
        f"<b>{team_pred['away_expected']:.2f}</b> para {away_team}, "
        f"sumando <b>{team_pred['total_expected']:.2f}</b>."
    )
    elems.append(Paragraph(resumen_txt, styles["BodyText"]))
    elems.append(Spacer(1, 10))

    def add_table(title: str, rows: list[list[str]]) -> None:
        elems.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        t = Table([["Indicador", "Valor"]] + rows, colWidths=[7.1 * cm, 10.2 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ]
            )
        )
        elems.append(t)
        elems.append(Spacer(1, 8))

    add_table(
        "1) Prediccion principal",
        [
            ["Faltas totales esperadas (modelo total)", _fmt(float(pred.expected_fouls))],
            ["Faltas esperadas local", _fmt(float(team_pred["home_expected"]))],
            ["Faltas esperadas visitante", _fmt(float(team_pred["away_expected"]))],
            ["Total reconciliado", _fmt(float(team_pred["total_expected"]))],
            ["Probabilidad arbitro estricto", f"{float(pred.referee_strict_prob):.1%}"],
            ["Pesos del ensemble (NB / Regresion / ANFIS)", f"{pred.weights[0]:.3f} / {pred.weights[1]:.3f} / {pred.weights[2]:.3f}"],
        ],
    )

    add_table(
        "2) Perfil arbitral",
        [
            ["Partidos historicos del arbitro", str(len(ref_all))],
            ["Media total de faltas con este arbitro", _fmt(ref_all_mean)],
            [f"Media con {home_team}", _fmt(ref_home_mean)],
            [f"Media con {away_team}", _fmt(ref_away_mean)],
            ["Media de faltas en sus ultimos 5", _fmt(ref_recent5_mean)],
            ["Lectura de forma de arbitraje", trend],
        ],
    )

    add_table(
        "3) Racha disciplinaria de equipos (ultimos 5 partidos)",
        [
            [f"{home_team} faltas cometidas (media)", _fmt(home_committed)],
            [f"{home_team} faltas recibidas (media)", _fmt(home_suffered)],
            [f"{away_team} faltas cometidas (media)", _fmt(away_committed)],
            [f"{away_team} faltas recibidas (media)", _fmt(away_suffered)],
        ],
    )

    add_table(
        "4) Historial H2H de faltas",
        [
            ["N partidos H2H usados", str(len(h2h))],
            ["Media H2H de faltas totales", _fmt(h2h_mean)],
            ["Rango H2H observado", f"{_fmt(min(h2h_totals) if h2h_totals else None)} - {_fmt(max(h2h_totals) if h2h_totals else None)}"],
        ],
    )

    add_table(
        "5) Contexto esperado del partido",
        [
            ["Tipo de partido (intensidad esperada)", intensidad],
            ["Riesgo disciplinario", riesgo],
            ["Es derby", "Si" if derby else "No"],
            ["Posesion esperada", f"{home_team}: {poss_home:.1f}% | {away_team}: {poss_away:.1f}%"],
            ["xFouls esperadas", f"{home_team}: {xf_home:.2f} | {away_team}: {xf_away:.2f} | total {xf_home + xf_away:.2f}"],
        ],
    )

    add_table("6) Senales de mercado", _market_readable(feat))

    nota = (
        "Nota de lectura: las estimaciones son probabilisticas, no deterministas. "
        "El valor central (media esperada) puede diferir del resultado real de un partido concreto "
        "por ruido arbitral, contexto tactico no observable y eventos de juego."
    )
    elems.append(Paragraph(nota, styles["Italic"]))

    doc.build(elems)
    return out_path


def main() -> int:
    home = "Villarreal"
    away = "Sociedad"
    md = 29
    out = build_report(home, away, md)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

