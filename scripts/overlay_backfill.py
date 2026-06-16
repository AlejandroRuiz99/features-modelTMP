"""
scripts/overlay_backfill.py — Backfill overlay predictions against historical actuals.

Usage:
    python scripts/overlay_backfill.py \\
        --narratives overlay/backfill/narratives \\
        --actuals overlay/backfill/actuals.json \\
        --output overlay/backfill/initial_calibration_report.md

For each narrative YAML found in --narratives:
  1. Load the narrative via overlay.loader.
  2. Re-run the ensemble prediction for the match (P3 overlay applied).
  3. Compute pre-overlay and post-overlay expected fouls.
  4. Look up actual fouls from --actuals (match key: "{Home}_vs_{Away}_{date}").
  5. Emit a markdown row:
       | match | pre_pred | post_pred | actual | line_pre | line_post | hit_pre | hit_post | rules_fired |

Missing actuals → NA.
Missing/invalid narratives → skipped, listed at end of report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path setup — must mirror run_prediction.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features_generator"
PRED_DIR = ROOT / "prediction_models"
sys.path.insert(0, str(PRED_DIR))
sys.path.insert(0, str(FEATURES_DIR))
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from overlay.loader import load_narrative  # noqa: E402, I001
from overlay.schema import Narrative  # noqa: E402


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class BackfillRow:
    """Result row for one match."""

    def __init__(
        self,
        match_key: str,
        pre_pred: float | None,
        post_pred: float | None,
        actual: float | None,
        line_pre: str,
        line_post: str,
        hit_pre: str,
        hit_post: str,
        rules_fired: str,
    ) -> None:
        self.match_key = match_key
        self.pre_pred = pre_pred
        self.post_pred = post_pred
        self.actual = actual
        self.line_pre = line_pre
        self.line_post = line_post
        self.hit_pre = hit_pre
        self.hit_post = hit_post
        self.rules_fired = rules_fired


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------


def _make_match_key(narrative: Narrative) -> str:
    """Build the canonical actuals lookup key for a narrative."""
    home = narrative.match.home.replace(" ", "_")
    away = narrative.match.away.replace(" ", "_")
    date = narrative.match.date
    return f"{home}_vs_{away}_{date}"


def _load_state() -> dict[str, Any] | None:
    """Load historical state from Supabase cache.

    Returns None if state cannot be loaded (offline, credentials missing, etc.).
    """
    try:
        from core.state_cache import get_state  # type: ignore[import]

        return get_state(refresh=False)
    except Exception as exc:
        print(
            f"[backfill] WARNING: Could not load Supabase state: {exc}", file=sys.stderr
        )
        return None


def _load_ensemble() -> Any | None:
    """Load the trained ensemble from checkpoints.

    Returns None if checkpoints not found.
    """
    try:
        from src.models.ensemble import FoulPredictionEnsemble  # type: ignore[import]

        checkpoint_dir = PRED_DIR / "checkpoints" / "ensemble"
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            print(
                f"[backfill] WARNING: Ensemble config not found: {config_path}",
                file=sys.stderr,
            )
            return None
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        ensemble = FoulPredictionEnsemble(config=cfg)
        ensemble.load(checkpoint_dir)
        return ensemble
    except Exception as exc:
        print(f"[backfill] WARNING: Could not load ensemble: {exc}", file=sys.stderr)
        return None


def _try_run_prediction(
    narrative: Narrative,
    state: dict[str, Any] | None,
    ensemble: Any | None,
) -> dict[str, Any] | None:
    """Attempt to run the ensemble prediction for the match in the narrative.

    Args:
        narrative: Parsed narrative dataclass.
        state: Pre-loaded Supabase state (or None if unavailable).
        ensemble: Pre-loaded ensemble (or None if unavailable).

    Returns:
        Dict with keys: pre_pred, post_pred, rules_fired (list of str).
        Returns None if prediction cannot be run (missing data, no Supabase, etc.).
    """
    if state is None or ensemble is None:
        return None

    try:
        from assembly import build_features  # type: ignore[import]
        from overlay.applier import apply_overlay
        from overlay.objective import inject_objectives_into_state
        from overlay.rules import load_catalog

        # P1: apply objective override BEFORE feature generation
        feat_state = state
        if narrative.objectives:
            feat_state = inject_objectives_into_state(state, narrative)

        # Build base feature dict for the match
        feat = build_features(
            state=feat_state,
            equipo_local_input=narrative.match.home,
            equipo_visitante_input=narrative.match.away,
            jornada=narrative.match.jornada,
            fecha_partido_input=narrative.match.date,
            arbitraje_source="not_available",
        )

        # Predict pre-overlay (model without narrative tilt)
        pred_raw = ensemble.predict(feat)
        pre_pred = round(float(pred_raw.expected_fouls), 2)

        # Build prediction dict for overlay (P3 needs pmf_total)
        prediction_for_overlay: dict[str, Any] = {
            "pmf_total": pred_raw.pmf_total,
            "expected_fouls": float(pred_raw.expected_fouls),
            "home_expected": float(pred_raw.expected_fouls * 0.55),
            "away_expected": float(pred_raw.expected_fouls * 0.45),
            "over_under": pred_raw.over_under or {},
        }

        # P3/P4: apply overlay
        catalog = load_catalog(ROOT / "overlay" / "rules.yaml")
        overlay_result = apply_overlay(prediction_for_overlay, narrative, catalog)
        post_pred = round(float(overlay_result.post_pmf_summary["mean"]), 2)
        rules_list = [r["id"] for r in overlay_result.rules_fired]

        return {
            "pre_pred": pre_pred,
            "post_pred": post_pred,
            "rules_fired": rules_list,
        }

    except Exception as exc:
        print(f"[backfill] WARNING: Could not run prediction: {exc}", file=sys.stderr)
        return None


def _hit_indicator(pred: float | None, actual: float | None, line: float) -> str:
    """Return HIT/MISS/NA for a prediction vs actual at a given line."""
    if pred is None or actual is None:
        return "NA"
    predicted_over = pred > line
    actual_over = actual > line
    return "HIT" if predicted_over == actual_over else "MISS"


def _best_line(pred: float | None) -> float:
    """Return a simple reference line close to the prediction."""
    if pred is None:
        return 25.5
    # Round to nearest 0.5
    return round(pred * 2) / 2


def _fmt(value: float | None, na: str = "NA") -> str:
    if value is None:
        return na
    return str(value)


def _process_narrative(
    stem: str,
    narrative: Narrative,
    actuals: dict[str, Any],
    state: dict[str, Any] | None = None,
    ensemble: Any | None = None,
) -> BackfillRow:
    """Process one narrative and return a BackfillRow."""
    match_key = _make_match_key(narrative)
    actual_raw = actuals.get(match_key)
    actual: float | None = float(actual_raw) if actual_raw is not None else None

    result = _try_run_prediction(narrative, state, ensemble)

    if result is None:
        # Prediction failed — use NA for everything
        return BackfillRow(
            match_key=f"{narrative.match.home} vs {narrative.match.away} ({narrative.match.date})",
            pre_pred=None,
            post_pred=None,
            actual=actual,
            line_pre="NA",
            line_post="NA",
            hit_pre="NA",
            hit_post="NA",
            rules_fired="",
        )

    pre_pred = result["pre_pred"]
    post_pred = result["post_pred"]
    rules_list: list[str] = result["rules_fired"]

    line = _best_line(pre_pred)

    hit_pre = _hit_indicator(pre_pred, actual, line)
    hit_post = _hit_indicator(post_pred, actual, line)

    return BackfillRow(
        match_key=f"{narrative.match.home} vs {narrative.match.away} ({narrative.match.date})",
        pre_pred=pre_pred,
        post_pred=post_pred,
        actual=actual,
        line_pre=str(line),
        line_post=str(line),
        hit_pre=hit_pre,
        hit_post=hit_post,
        rules_fired=", ".join(rules_list) if rules_list else "—",
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

HEADER = (
    "| match | pre_pred | post_pred | actual | line_pre | line_post "
    "| hit_pre | hit_post | rules_fired |"
)
SEPARATOR = (
    "|-------|----------|-----------|--------|----------|-----------|"
    "---------|----------|-------------|"
)


def _row_to_md(row: BackfillRow) -> str:
    return (
        f"| {row.match_key} "
        f"| {_fmt(row.pre_pred)} "
        f"| {_fmt(row.post_pred)} "
        f"| {_fmt(row.actual)} "
        f"| {row.line_pre} "
        f"| {row.line_post} "
        f"| {row.hit_pre} "
        f"| {row.hit_post} "
        f"| {row.rules_fired} |"
    )


def render_report(
    rows: list[BackfillRow],
    skipped: list[str],
    narratives_dir: str,
    actuals_path: str,
) -> str:
    """Render the full markdown report."""
    lines: list[str] = [
        "# Overlay Backfill Calibration Report",
        "",
        f"**Narratives dir**: `{narratives_dir}`  ",
        f"**Actuals file**: `{actuals_path}`  ",
        f"**Rows**: {len(rows)}  ",
        "",
        HEADER,
        SEPARATOR,
    ]
    for row in rows:
        lines.append(_row_to_md(row))
    lines.append("")

    if skipped:
        lines += [
            "## Skipped narratives",
            "",
            "The following narrative files were skipped due to load/parse errors:",
            "",
        ]
        for s in skipped:
            lines.append(f"- `{s}`")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill overlay predictions against historical actuals."
    )
    parser.add_argument(
        "--narratives",
        required=True,
        help="Directory containing narrative YAML files, or space-separated list of files.",
    )
    parser.add_argument(
        "--actuals",
        required=True,
        help="JSON file mapping match key → actual_fouls (int|float). Use null for unknown.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output markdown file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load actuals
    actuals_path = Path(args.actuals)
    if not actuals_path.exists():
        print(
            f"[backfill] ERROR: actuals file not found: {actuals_path}", file=sys.stderr
        )
        sys.exit(1)

    actuals_raw: dict = json.loads(actuals_path.read_text(encoding="utf-8"))
    # Filter out null values
    actuals: dict[str, Any] = {k: v for k, v in actuals_raw.items() if v is not None}

    # Load narratives
    narratives_dir = Path(args.narratives)
    skipped: list[str] = []
    loaded_narratives: dict[str, Narrative] = {}

    if narratives_dir.is_dir():
        for yaml_path in sorted(narratives_dir.glob("*.yaml")):
            try:
                narr = load_narrative(yaml_path)
                loaded_narratives[yaml_path.stem] = narr
            except Exception as exc:
                print(
                    f"[backfill] Skipping {yaml_path.name}: {exc}",
                    file=sys.stderr,
                )
                skipped.append(yaml_path.name)
    else:
        print(
            f"[backfill] ERROR: --narratives path is not a directory: {narratives_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load state and ensemble once (shared across all matches)
    print("[backfill] Loading Supabase state...", file=sys.stderr)
    state = _load_state()
    if state is None:
        print(
            "[backfill] WARNING: Supabase state unavailable — predictions will be NA",
            file=sys.stderr,
        )

    print("[backfill] Loading ensemble...", file=sys.stderr)
    ensemble = _load_ensemble()
    if ensemble is None:
        print(
            "[backfill] WARNING: Ensemble unavailable — predictions will be NA",
            file=sys.stderr,
        )

    # Process each narrative
    rows: list[BackfillRow] = []
    for stem, narrative in loaded_narratives.items():
        print(f"[backfill] Processing: {stem}", file=sys.stderr)
        row = _process_narrative(
            stem, narrative, actuals, state=state, ensemble=ensemble
        )
        rows.append(row)

    # Render report
    report = render_report(
        rows=rows,
        skipped=skipped,
        narratives_dir=str(narratives_dir),
        actuals_path=str(actuals_path),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(
        f"[backfill] Report written: {output_path} ({len(rows)} rows)", file=sys.stderr
    )


if __name__ == "__main__":
    main()
