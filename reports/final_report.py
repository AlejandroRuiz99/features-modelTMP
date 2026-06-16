"""
reports.final_report — Generate consolidated PDF report from a run folder.

Combines prediction_report.md + ev_report.md + manifest data into a single
PDF saved to {run_dir}/report/final_report.pdf.

Uses reportlab (pure-Python, no external deps). Reads:
  - {run_dir}/manifest.yaml
  - {run_dir}/input/matches.yaml
  - {run_dir}/input/narratives/*.yaml
  - {run_dir}/prediction/prediction.json
  - {run_dir}/ev/ev_table.json (optional)
  - {run_dir}/ev/team_probs_post_overlay.json (optional)
  - {run_dir}/odds/manual_odds_*.json or codere_snapshot.json (optional)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

__all__ = ["generate_final_report"]


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles["title"] = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=12,
        textColor=colors.HexColor("#0b2545"),
    )
    styles["subtitle"] = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=10,
        textColor=colors.HexColor("#445566"),
    )
    styles["h1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#0b2545"),
    )
    styles["h2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#1f3c5b"),
    )
    styles["body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    styles["small"] = ParagraphStyle(
        "Small",
        parent=base["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#666"),
    )
    styles["mono"] = ParagraphStyle(
        "Mono",
        parent=base["Code"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#222"),
    )
    return styles


def _table_style(highlight_header: bool = True) -> TableStyle:
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#0b2545")),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#f4f7fb")],
        ),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if highlight_header:
        cmds.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")))
    return TableStyle(cmds)


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------


def _load_run_data(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"run_dir": str(run_dir)}

    manifest_path = run_dir / "manifest.yaml"
    data["manifest"] = (
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    matches_path = run_dir / "input" / "matches.yaml"
    data["matches"] = (
        yaml.safe_load(matches_path.read_text(encoding="utf-8"))
        if matches_path.exists()
        else {}
    )

    narratives_dir = run_dir / "input" / "narratives"
    narratives: dict[str, Any] = {}
    if narratives_dir.is_dir():
        for nf in sorted(narratives_dir.glob("*.yaml")):
            try:
                narratives[nf.stem] = yaml.safe_load(nf.read_text(encoding="utf-8"))
            except Exception:
                continue
    data["narratives"] = narratives

    pred_path = run_dir / "prediction" / "prediction.json"
    data["predictions"] = (
        json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else []
    )

    ev_table_path = run_dir / "ev" / "ev_table.json"
    data["ev_table"] = (
        json.loads(ev_table_path.read_text(encoding="utf-8"))
        if ev_table_path.exists()
        else None
    )

    team_probs_path = run_dir / "ev" / "team_probs_post_overlay.json"
    data["team_probs"] = (
        json.loads(team_probs_path.read_text(encoding="utf-8"))
        if team_probs_path.exists()
        else None
    )

    # Find any odds file
    odds_dir = run_dir / "odds"
    data["odds"] = None
    if odds_dir.is_dir():
        for of in sorted(odds_dir.glob("*.json")):
            try:
                data["odds"] = json.loads(of.read_text(encoding="utf-8"))
                data["odds_source_file"] = of.name
                break
            except Exception:
                continue

    return data


# -----------------------------------------------------------------------------
# Section builders
# -----------------------------------------------------------------------------


def _section_header(data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list:
    manifest = data["manifest"]
    matches_yaml = data["matches"]
    jornada = matches_yaml.get("jornada", "?")
    matches = matches_yaml.get("matches", []) or []

    flow: list = []
    flow.append(
        Paragraph(f"Foul Prediction Report — Jornada {jornada}", styles["title"])
    )

    sub_lines = []
    sub_lines.append(f"<b>Run ID:</b> {manifest.get('run_id', '?')}")
    sub_lines.append(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if manifest.get("git_commit"):
        sub_lines.append(f"<b>Git commit:</b> {manifest['git_commit'][:12]}")
    sub_lines.append(f"<b>Seed:</b> {manifest.get('seed', '?')}")
    sub_lines.append(f"<b>Stats freshness:</b> {manifest.get('stats_freshness', '?')}")
    if manifest.get("bankroll"):
        sub_lines.append(f"<b>Bankroll:</b> €{manifest['bankroll']:.2f}")
    flow.append(Paragraph("<br/>".join(sub_lines), styles["subtitle"]))

    if matches:
        match_lines = ["<b>Matches in this run:</b>"]
        for m in matches:
            match_lines.append(
                f"&nbsp;&nbsp;• {m.get('home', '?')} vs {m.get('away', '?')} — "
                f"{m.get('date', '?')} — Ref: {m.get('referee') or '?'}"
            )
        flow.append(Paragraph("<br/>".join(match_lines), styles["body"]))
    flow.append(Spacer(1, 6))
    return flow


def _section_match(
    pred: dict[str, Any],
    narrative: dict[str, Any] | None,
    team_probs: dict[str, Any] | None,
    ev_table: dict[str, Any] | None,
    odds: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
) -> list:
    home = pred.get("match", "?").split(" vs ")[0]
    away = pred.get("match", "?").split(" vs ")[-1]
    date = pred.get("date", "?")
    jornada = pred.get("jornada", "?")
    referee = pred.get("referee", "?")

    flow: list = []
    flow.append(Paragraph(f"{home} vs {away}", styles["h1"]))
    flow.append(
        Paragraph(
            f"<b>Date:</b> {date} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Jornada:</b> {jornada} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Referee:</b> {referee}",
            styles["body"],
        )
    )
    flow.append(Spacer(1, 4))

    # ===== Prediction summary =====
    flow.append(Paragraph("Prediction summary", styles["h2"]))

    overlay = pred.get("overlay") or {}
    pre_total = overlay.get("pre_expected_fouls", pred.get("expected_fouls", 0.0))
    post_total = overlay.get("post_expected_fouls", pred.get("total_expected", 0.0))
    delta_applied = overlay.get("delta_fouls_applied", 0.0)
    var_scale = overlay.get("variance_scale_applied", 1.0)
    kelly_scale = overlay.get("kelly_scale_applied", 1.0)

    summary_rows = [
        ["Metric", "Pre-overlay", "Post-overlay", "Δ"],
        [
            "Total fouls",
            f"{pre_total:.2f}",
            f"{post_total:.2f}",
            f"{delta_applied:+.2f}",
        ],
        ["Home expected", "—", f"{pred.get('home_expected', 0):.2f}", "—"],
        ["Away expected", "—", f"{pred.get('away_expected', 0):.2f}", "—"],
        ["Variance scale", "1.00", f"×{var_scale:.2f}", "—"],
        ["Kelly scale", "1.00", f"×{kelly_scale:.2f}", "—"],
        [
            "Referee strict prob",
            "—",
            f"{pred.get('referee_strict_prob', 0) * 100:.1f}%",
            "—",
        ],
    ]
    t = Table(summary_rows, colWidths=[5 * cm, 3.2 * cm, 3.2 * cm, 2.5 * cm])
    t.setStyle(_table_style())
    flow.append(t)
    flow.append(Spacer(1, 6))

    # ===== Overlay rules =====
    rules_fired = overlay.get("rules_fired") or []
    if rules_fired:
        flow.append(Paragraph("Overlay rules fired", styles["h2"]))
        rule_rows = [["Rule ID", "Direction", "Δ applied", "Suppressed"]]
        for r in rules_fired:
            rule_rows.append(
                [
                    r.get("id", "?"),
                    r.get("direction", "?"),
                    f"{r.get('magnitude_applied', 0):+.2f}",
                    "yes" if r.get("suppressed_by_floor") else "no",
                ]
            )
        rt = Table(rule_rows, colWidths=[7 * cm, 3 * cm, 2.5 * cm, 2.5 * cm])
        rt.setStyle(_table_style())
        flow.append(rt)
        flow.append(Spacer(1, 6))

    # ===== Narrative summary =====
    if narrative:
        flow.append(Paragraph("Narrative context", styles["h2"]))
        objs = narrative.get("objectives") or {}
        stakes = narrative.get("stakes") or {}
        flags = narrative.get("special_flags") or []

        narr_rows = [
            ["Side", "Objective", "Urgency", "Stakes"],
        ]
        for side in ("home", "away"):
            obj = objs.get(side) or {}
            narr_rows.append(
                [
                    side.upper(),
                    str(obj.get("label", "?")),
                    f"{obj.get('urgency_base', '—')}"
                    if obj.get("urgency_base") is not None
                    else "—",
                    str(stakes.get(side, "—")),
                ]
            )
        nt = Table(narr_rows, colWidths=[2 * cm, 4.5 * cm, 3 * cm, 2.5 * cm])
        nt.setStyle(_table_style())
        flow.append(nt)
        flow.append(Spacer(1, 4))

        ctx_lines = []
        if narrative.get("intensity_override") is not None:
            ctx_lines.append(
                f"<b>Intensity override:</b> {narrative['intensity_override']}"
            )
        if narrative.get("physicality_bias") is not None:
            ctx_lines.append(
                f"<b>Physicality bias:</b> {narrative['physicality_bias']:+d}"
            )
        if narrative.get("referee_factor") is not None:
            ctx_lines.append(f"<b>Referee factor:</b> {narrative['referee_factor']:+d}")
        if narrative.get("confidence_level") is not None:
            ctx_lines.append(f"<b>Confidence:</b> {narrative['confidence_level']}/5")
        if flags:
            ctx_lines.append(f"<b>Flags:</b> {', '.join(flags)}")
        if ctx_lines:
            flow.append(
                Paragraph(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(ctx_lines), styles["body"])
            )
        if narrative.get("notes"):
            flow.append(Spacer(1, 2))
            flow.append(Paragraph(f"<i>{narrative['notes']}</i>", styles["small"]))
        flow.append(Spacer(1, 6))

    # ===== O/U lines =====
    ou = pred.get("over_under") or {}
    if ou:
        flow.append(
            Paragraph(
                "Total fouls — Over/Under probabilities (post-overlay)", styles["h2"]
            )
        )
        # Pick a curated set of lines around the post-overlay mean
        post_mu = float(post_total)
        candidate_lines = sorted(float(k) for k in ou.keys())
        # Show 4 below and 4 above the mean
        lines_to_show = [l for l in candidate_lines if abs(l - post_mu) <= 5.0][:9]
        ou_rows = [["Line", "OVER %", "UNDER %"]]
        for line in lines_to_show:
            row = ou[str(line)]
            ou_rows.append(
                [
                    f"{line:.1f}",
                    f"{row['over'] * 100:.1f}%",
                    f"{row['under'] * 100:.1f}%",
                ]
            )
        ot = Table(ou_rows, colWidths=[3 * cm, 4 * cm, 4 * cm])
        ot.setStyle(_table_style())
        flow.append(ot)
        flow.append(Spacer(1, 6))

    # ===== EV table for this match =====
    if ev_table and ev_table.get("picks"):
        flow.append(Paragraph("EV analysis (markets vs offered odds)", styles["h2"]))
        ev_rows = [
            [
                "Market",
                "Side",
                "Line",
                "Model %",
                "Odds",
                "Implied %",
                "Edge",
                "Stake (€)",
                "Verdict",
            ]
        ]
        for p in ev_table["picks"]:
            ev_rows.append(
                [
                    p["market"],
                    p["side"],
                    f"{p['line']:.1f}",
                    f"{p['model_prob'] * 100:.1f}%",
                    f"{p['odds']:.2f}",
                    f"{p['implied_prob'] * 100:.1f}%",
                    p["edge_pct"],
                    f"€{p['euros']:.2f}",
                    p["verdict"],
                ]
            )
        evt = Table(
            ev_rows,
            colWidths=[
                3.2 * cm,
                1.4 * cm,
                1.4 * cm,
                1.6 * cm,
                1.4 * cm,
                1.6 * cm,
                1.6 * cm,
                1.6 * cm,
                1.6 * cm,
            ],
        )
        evt.setStyle(_table_style())
        # Highlight rows by verdict
        for i, p in enumerate(ev_table["picks"], start=1):
            if p["verdict"] == "BET":
                evt.setStyle(
                    TableStyle(
                        [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#d4ecd8"))]
                    )
                )
            elif p.get("edge", 0) > 0:
                evt.setStyle(
                    TableStyle(
                        [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff4d4"))]
                    )
                )
        flow.append(evt)
        flow.append(Spacer(1, 4))

        # Summary line
        n_bets = sum(1 for p in ev_table["picks"] if p["verdict"] == "BET")
        total_stake = sum(p["euros"] for p in ev_table["picks"])
        bankroll = ev_table.get("bankroll", 0)
        flow.append(
            Paragraph(
                f"<b>Recommended bets:</b> {n_bets} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Total stake:</b> €{total_stake:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Bankroll:</b> €{bankroll:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Kelly:</b> ×{ev_table.get('kelly_scale', 1.0):.2f}",
                styles["body"],
            )
        )
    elif odds:
        flow.append(
            Paragraph("Odds available but EV table not produced.", styles["body"])
        )
    else:
        flow.append(
            Paragraph("No odds available — EV analysis skipped.", styles["body"])
        )

    flow.append(Spacer(1, 8))
    return flow


def _section_footer(data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list:
    flow: list = []
    flow.append(Paragraph("Run artifacts", styles["h2"]))
    run_dir = Path(data["run_dir"])
    files = []
    for sub in ("input", "features", "prediction", "ev", "odds", "report"):
        sub_path = run_dir / sub
        if sub_path.is_dir():
            for f in sorted(sub_path.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(run_dir)
                    files.append(str(rel).replace("\\", "/"))
    flow.append(
        Paragraph("<br/>".join(f"&nbsp;&nbsp;{x}" for x in files[:60]), styles["mono"])
    )
    flow.append(Spacer(1, 4))
    flow.append(
        Paragraph(
            "Generated by reports.final_report — predecir-jornada-v2 pipeline.",
            styles["small"],
        )
    )
    return flow


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def generate_final_report(
    run_dir: Path | str, output_path: Path | str | None = None
) -> Path:
    """Generate consolidated PDF report for a completed run.

    Args:
        run_dir: Path to the run folder (e.g. runs/2026-04-30_j34-vie).
        output_path: Optional explicit output path. Defaults to
            {run_dir}/report/final_report.pdf.

    Returns:
        Path to the generated PDF.

    Requires: at least prediction.json. EV section is optional.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    data = _load_run_data(run_dir)

    if not data["predictions"]:
        raise ValueError(
            f"No prediction.json found in {run_dir}/prediction/ — cannot generate final report"
        )

    if output_path is None:
        output_path = run_dir / "report" / "final_report.pdf"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"Foul Prediction — {data['manifest'].get('run_id', 'run')}",
    )

    story: list = []
    story.extend(_section_header(data, styles))

    # Per-match sections
    narratives = data.get("narratives") or {}
    ev_table = data.get("ev_table")
    team_probs = data.get("team_probs")
    odds = data.get("odds")
    predictions = data["predictions"]

    for i, pred in enumerate(predictions):
        # Find matching narrative
        match_str = pred.get("match", "")
        date = pred.get("date", "")
        narrative = None
        if match_str and date:
            home, away = (match_str.split(" vs ", 1) + [""])[:2]
            slug = f"{home.lower()}_vs_{away.lower()}_{date}"
            narrative = narratives.get(slug) or narratives.get(slug.replace(" ", ""))
            if not narrative:
                # try case-insensitive
                for k, v in narratives.items():
                    if k.lower() == slug.lower():
                        narrative = v
                        break

        # Filter EV picks for this specific match by match name
        ev_for_match = None
        if ev_table and ev_table.get("picks"):
            match_picks = [
                p
                for p in ev_table["picks"]
                if p.get("match", "").lower().replace(" ", "")
                in match_str.lower().replace(" ", "")
                or match_str.lower().replace(" ", "")
                in p.get("match", "").lower().replace(" ", "")
            ]
            if match_picks:
                ev_for_match = dict(ev_table)
                ev_for_match["picks"] = match_picks

        story.extend(
            _section_match(
                pred=pred,
                narrative=narrative,
                team_probs=team_probs,
                ev_table=ev_for_match,
                odds=odds,
                styles=styles,
            )
        )
        if i < len(predictions) - 1:
            story.append(PageBreak())

    story.append(Spacer(1, 8))
    story.extend(_section_footer(data, styles))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate final PDF report from a run folder."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run folder (e.g. runs/2026-04-30_j34-vie)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Explicit output path"
    )
    args = parser.parse_args()
    out = generate_final_report(args.run_dir, args.output)
    print(f"Final report written to: {out}")
