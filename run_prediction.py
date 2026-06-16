"""
run_prediction.py - Predice faltas para un partido de La Liga.

Uso (partido unico):
  python run_prediction.py --local "Real Madrid" --visitante "Ath Madrid" --jornada 29 --fecha 2026-03-22
  python run_prediction.py --local "Levante" --visitante "Oviedo" --jornada 29 --fecha 2026-03-21 --arbitro "Munuera Montero"

Modo batch (JSON o CSV):
  python run_prediction.py --batch-file partidos.json --output-json resultados.json

Flags adicionales:
  --features-profile  prediction | training | minimal
  --refresh-data      Fuerza recarga desde Supabase
  --output-json PATH  Guarda la prediccion en JSON
  --log-level         CRITICAL | ERROR | WARNING | INFO | DEBUG
  --narrative PATH    Ruta a YAML de narrativa (overlay P1/P3/P4 para un partido)
  --narratives DIR    Directorio de YAMLs de narrativa para modo batch
  --overlay-log-dir   Directorio donde escribir los logs de overlay (default: overlay/logs)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path: cada subdirectorio usa imports sin prefijo de paquete
# (ej: "from core.xxx" en features_generator, "from src.models.xxx" en prediction_models).
# Ambos directorios deben estar en sys.path ANTES de cualquier import del proyecto.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features_generator"
PRED_DIR = ROOT / "prediction_models"

sys.path.insert(0, str(PRED_DIR))  # habilita: from src.models.xxx, from src.utils.xxx
sys.path.insert(
    0, str(FEATURES_DIR)
)  # habilita: from core.xxx, from generate import, etc.

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
from generate import generate_features  # noqa: E402

from assembly import build_features  # noqa: E402
from src.models.ensemble import FoulPredictionEnsemble  # noqa: E402

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"


# ---------------------------------------------------------------------------
# _print_prediction: salida legible por pantalla
# ---------------------------------------------------------------------------


def _print_prediction(ensemble: FoulPredictionEnsemble, feat: dict[str, Any]) -> None:
    """Imprime la prediccion de faltas en formato legible."""
    home = feat["home_team"]
    away = feat["away_team"]
    referee = feat.get("referee", "Desconocido")
    matchday = feat.get("matchday", "?")
    season = feat.get("season", "")

    pred = ensemble.predict(feat)
    team_pred = (
        ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True) or {}
    )

    lines = [21.5, 23.5, 24.5, 25.5, 27.5, 29.5]

    print("=" * 65)
    print(f"  {home} vs {away} | J{matchday} {season}")
    print(f"  Arbitro: {referee}")
    print("=" * 65)

    print("\n--- Senales clave ---")
    print(
        f"  Arbitro GMM:        permisivo={feat['referee_mu_permisivo']:.1f} | "
        f"estricto={feat['referee_mu_estricto']:.1f} | "
        f"p(estricto)={feat['referee_peso_estricto']:.0%} | "
        f"n={feat['referee_n_partidos']}"
    )
    print(
        f"  xFaltas esperadas:  {feat['xfouls_home']:.1f} (L) + "
        f"{feat['xfouls_away']:.1f} (V) = "
        f"{feat['xfouls_home'] + feat['xfouls_away']:.1f}"
    )

    market_total = feat.get("foul_market_implied_mean", 0.0)
    market_prob_over = feat.get("foul_market_prob_over", 0.0)
    market_local = feat.get(
        "foul_market_local_implied_mean", market_total / 2 if market_total else 0.0
    )
    market_visitor = feat.get(
        "foul_market_vis_implied_mean", market_total / 2 if market_total else 0.0
    )
    print(
        f"  Mercado faltas OU:  total={market_total:.1f} "
        f"(prob_mas={market_prob_over:.1%}) | "
        f"local={market_local:.1f} | visitante={market_visitor:.1f}"
    )

    print(
        f"  Forma reciente:     {feat['forma_fouls_home']:.1f} faltas (L) / "
        f"{feat['forma_fouls_away']:.1f} faltas (V)"
    )
    print(
        f"  Agresividad (norm): {feat['aggressiveness_volume_home']:.2f} (L) / "
        f"{feat['aggressiveness_volume_away']:.2f} (V)"
    )
    print(
        f"  Urgencia:           {feat['urgency_home']:.2f} (L) / {feat['urgency_away']:.2f} (V)"
    )
    print(
        f"  Momentum:           {feat['momentum_home']:.3f} (L) / {feat['momentum_away']:.3f} (V)"
    )
    print(
        f"  Dias descanso:      {feat['days_rest_home']:.0f} (L) / {feat['days_rest_away']:.0f} (V)"
    )
    print(
        f"  Delta arbitro:      {feat['ref_home_delta']:+.2f} (L) / {feat['ref_away_delta']:+.2f} (V)"
    )
    print(
        f"  Derby: {feat['is_derby']} | "
        f"Season phase: {feat['season_phase']:.2f} | "
        f"Pace index: {feat['pace_index_curr']:.1f}"
    )
    print(
        f"  H2H: {feat['h2h_faltas_media']:.1f} faltas media ({feat['h2h_partidos']} partidos previos)"
    )
    print(
        f"  Intensidad: {feat['intensidad_esperada']} | "
        f"Riesgo disciplinario: {feat['riesgo_disciplinario']}"
    )

    print("\n--- Prediccion ensemble ---")
    print(f"  Faltas esperadas (raw):        {pred.expected_fouls:.2f}")
    print(f"  P(arbitro estricto):            {pred.referee_strict_prob * 100:.1f}%")
    print(
        f"  Pesos gating [NB, Reg, ANFIS]: "
        f"[{pred.weights[0]:.3f}, {pred.weights[1]:.3f}, {pred.weights[2]:.3f}]"
    )

    ou_table = pred.over_under
    if team_pred.get("reconciled") and team_pred.get("total_pmf") is not None:
        ou_table = team_pred["total_pmf"].over_under_table(lines)

    print("\n--- Over/Under ---")
    for line in lines:
        if line in ou_table:
            p_o, p_u = ou_table[line]
            tag = "  <-- linea mercado" if abs(line - market_total) < 1.0 else ""
            print(
                f"  OU {line:.1f}: Over={p_o * 100:5.1f}% | Under={p_u * 100:5.1f}%{tag}"
            )

    if team_pred:
        print("\n--- Desglose por equipo ---")
        print(f"  Total reconciliado:  {team_pred['total_expected']:.2f}")
        print(f"  {home:20s}: {team_pred['home_expected']:.2f}")
        print(f"  {away:20s}: {team_pred['away_expected']:.2f}")
        if team_pred.get("reconciled"):
            print(
                f"  Reconciliacion: raw({team_pred['raw_total_expected']:.2f}) "
                f"-> coherent({team_pred['total_expected']:.2f} = "
                f"{team_pred['home_expected']:.2f} + {team_pred['away_expected']:.2f})"
            )

    print("=" * 65)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _fuzzy_match_team(name: str, canonical: list[str]) -> str | None:
    from core.utils import TEAM_ALIASES, fuzzy_name_search

    return fuzzy_name_search(name, canonical, TEAM_ALIASES) or None


def _load_state(refresh: bool = False) -> dict[str, Any]:
    from core.state_cache import get_state

    return get_state(refresh=refresh)


def _load_ensemble() -> FoulPredictionEnsemble:
    config_path = CHECKPOINT_DIR / "config.json"
    regression_path = CHECKPOINT_DIR / "regression.pt"
    if not config_path.exists() or not regression_path.exists():
        logger.error(
            f"Checkpoint no encontrado en {CHECKPOINT_DIR}. "
            "Entrena primero con: python -m scripts.train"
        )
        sys.exit(1)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(CHECKPOINT_DIR)
    return ensemble


def _validate_single(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Valida inputs de modo single-match; muta args con nombres canonicos si hay match."""
    errors: list[str] = []

    if not _validate_date(args.fecha):
        errors.append(f"Fecha invalida '{args.fecha}'. Usa formato YYYY-MM-DD.")

    if not (1 <= args.jornada <= 38):
        errors.append(f"Jornada debe estar entre 1 y 38, recibido: {args.jornada}.")

    canonical = list(state.get("scores", {}).keys())
    if canonical:
        home_match = _fuzzy_match_team(args.local, canonical)
        away_match = _fuzzy_match_team(args.visitante, canonical)
        if not home_match:
            errors.append(
                f"Equipo local '{args.local}' no reconocido. "
                f"Prueba con: {canonical[:5]}..."
            )
        if not away_match:
            errors.append(
                f"Equipo visitante '{args.visitante}' no reconocido. "
                f"Prueba con: {canonical[:5]}..."
            )
        args.local = home_match or args.local
        args.visitante = away_match or args.visitante
    else:
        logger.warning(
            "No se pudo obtener la lista de equipos; se omite validacion de nombres."
        )

    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)


def _prediction_to_dict(
    ensemble: FoulPredictionEnsemble,
    feat: dict[str, Any],
    narrative: Any | None = None,
    catalog: list[Any] | None = None,
    overlay_log_dir: Path | None = None,
) -> dict[str, Any]:
    """Ejecuta la prediccion y devuelve un dict JSON-serializable.

    If ``narrative`` is provided (and ``catalog`` is not None), applies the
    overlay pipeline (P3 + P4) after ensemble.predict() and adds an ``overlay``
    section to the output dict.  A JSON log is written to ``overlay_log_dir``
    (default: overlay/logs/).  The prediction JSON (non-overlay fields) remains
    byte-identical to the no-narrative case.
    """
    pred = ensemble.predict(feat)
    team_pred = (
        ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True) or {}
    )

    lines = [21.5, 23.5, 24.5, 25.5, 27.5, 29.5]
    ou_table = pred.over_under
    if team_pred.get("reconciled") and team_pred.get("total_pmf") is not None:
        ou_table = team_pred["total_pmf"].over_under_table(lines)

    result: dict[str, Any] = {
        "match": f"{feat['home_team']} vs {feat['away_team']}",
        "date": feat.get("date") or "",
        "jornada": feat.get("matchday") or 0,
        "referee": feat.get("referee") or "",
        "expected_fouls": float(pred.expected_fouls),
        "referee_strict_prob": float(pred.referee_strict_prob),
        "weights": [float(w) for w in pred.weights],
        "home_expected": float(team_pred.get("home_expected", 0.0)),
        "away_expected": float(team_pred.get("away_expected", 0.0)),
        "total_expected": float(team_pred.get("total_expected", pred.expected_fouls)),
        "reconciled": bool(team_pred.get("reconciled", False)),
        "over_under": {
            str(k): {"over": float(v[0]), "under": float(v[1])}
            for k, v in (pred.over_under or {}).items()
        },
    }

    # Overlay P3 + P4 (only when narrative is supplied)
    if narrative is not None and catalog is not None:
        result = _apply_overlay_post_prediction(
            base_result=result,
            pred_raw=pred,
            team_pred=team_pred,
            ou_table=ou_table,
            narrative=narrative,
            catalog=catalog,
            overlay_log_dir=overlay_log_dir,
        )

    return result


# ---------------------------------------------------------------------------
# Overlay helpers (P3 + P4 post-prediction wiring)
# ---------------------------------------------------------------------------


def _apply_overlay_post_prediction(
    base_result: dict[str, Any],
    pred_raw: Any,
    team_pred: dict[str, Any],
    ou_table: dict,
    narrative: Any,
    catalog: list[Any],
    overlay_log_dir: Path | None,
) -> dict[str, Any]:
    """Apply overlay P3 (PMF tilt) and P4 (kelly scale) after ensemble.predict().

    Adds an 'overlay' key to the result dict with a summary of what was applied.
    Writes a JSON log to overlay_log_dir (default: overlay/logs/).
    NEVER modifies the core prediction fields (identity when no rules fire).
    """
    from datetime import datetime, timezone

    from overlay.applier import apply_overlay
    from overlay.log_writer import build_log_overlay_section, write_overlay_log

    # Build a prediction dict for applier (needs pmf_total, expected_fouls, etc.)
    prediction_for_overlay: dict[str, Any] = {
        "pmf_total": pred_raw.pmf_total,
        "expected_fouls": float(pred_raw.expected_fouls),
        "home_expected": float(
            team_pred.get("home_expected", pred_raw.expected_fouls * 0.55)
        ),
        "away_expected": float(
            team_pred.get("away_expected", pred_raw.expected_fouls * 0.45)
        ),
        "over_under": ou_table,
    }

    overlay_result = apply_overlay(prediction_for_overlay, narrative, catalog)

    # Build overlay summary for output JSON
    overlay_summary: dict[str, Any] = {
        "rules_fired": overlay_result.rules_fired,
        "delta_fouls_applied": overlay_result.aggregated_effect.delta_fouls,
        "variance_scale_applied": overlay_result.aggregated_effect.variance_scale,
        "kelly_scale_applied": overlay_result.aggregated_effect.kelly_scale,
        "pre_expected_fouls": overlay_result.pre_pmf_summary["mean"],
        "post_expected_fouls": overlay_result.post_pmf_summary["mean"],
        "suppressed_by_floor_count": overlay_result.suppressed_by_floor_count,
    }

    # Write overlay log
    log_dir = overlay_log_dir or (ROOT / "overlay" / "logs")
    # Best-effort narrative raw text (we only have the parsed object here)
    narrative_raw = f"# auto-generated by run_prediction\nconfidence_level: {narrative.confidence_level}\n"

    log_record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "match": {
            "home": narrative.match.home,
            "away": narrative.match.away,
            "date": narrative.match.date,
        },
        "narrative_raw": narrative_raw,
        "parsed_flags": {
            "confidence_level": narrative.confidence_level,
            "special_flags": narrative.special_flags or [],
            "objectives": {
                side: {"label": oo.label, "urgency_base": oo.urgency_base}
                for side, oo in (narrative.objectives or {}).items()
            },
            "stakes": (
                {"home": narrative.stakes.home, "away": narrative.stakes.away}
                if narrative.stakes
                else None
            ),
            "rotations": narrative.rotations,
            "intensity_override": narrative.intensity_override,
            "physicality_bias": narrative.physicality_bias,
            "referee_factor": narrative.referee_factor,
        },
        "pre_overlay": build_log_overlay_section(
            {
                "expected_fouls": overlay_result.pre_pmf_summary["mean"],
                "pmf_summary": overlay_result.pre_pmf_summary,
            }
        ),
        "rules_fired": overlay_result.rules_fired,
        "post_overlay": build_log_overlay_section(
            {
                "expected_fouls": overlay_result.post_pmf_summary["mean"],
                "pmf_summary": overlay_result.post_pmf_summary,
            }
        ),
        "kelly_raw_vs_scaled": {
            "kelly_raw": overlay_result.kelly_raw,
            "kelly_scaled": overlay_result.kelly_scaled,
        },
        "actual_fouls": None,
    }

    try:
        log_path = write_overlay_log(log_record, Path(log_dir))
        overlay_summary["log"] = str(log_path)
        logger.info(
            f"Overlay: {len(overlay_result.rules_fired)} rules fired "
            f"({overlay_result.aggregated_effect.delta_fouls:+.2f} fouls, "
            f"var x{overlay_result.aggregated_effect.variance_scale:.2f}, "
            f"kelly x{overlay_result.aggregated_effect.kelly_scale:.2f}), "
            f"log: {log_path}"
        )
    except Exception as exc:
        logger.warning(f"Could not write overlay log: {exc}")
        overlay_summary["log"] = None

    result = dict(base_result)
    result["overlay"] = overlay_summary
    return result


def _find_narrative_for_match(
    narratives_dir: Path,
    home: str,
    away: str,
    date: str,
) -> Path | None:
    """Find a narrative YAML file for the given match in the narratives directory.

    Tries two naming conventions in order:
    1. Exact: {home}_vs_{away}_{date}.yaml  (with accents/spaces)
    2. Slugified: {slug(home)}_vs_{slug(away)}_{date}.yaml  (lowercase, no accents/spaces)

    Returns None if not found or directory doesn't exist.
    """
    import unicodedata
    import re as _re

    if not narratives_dir.is_dir():
        return None

    # Primary: exact match (works on case-insensitive filesystems like Windows)
    candidate = narratives_dir / f"{home}_vs_{away}_{date}.yaml"
    if candidate.is_file():
        return candidate

    # Fallback: slugified names (lowercase, strip accents, remove non-alphanumeric)
    def _slugify(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower()
        s = _re.sub(r"[^a-z0-9]+", "", s)
        return s

    slug_candidate = (
        narratives_dir / f"{_slugify(home)}_vs_{_slugify(away)}_{date}.yaml"
    )
    if slug_candidate.is_file():
        return slug_candidate

    return None


# ---------------------------------------------------------------------------
# Modo batch
# ---------------------------------------------------------------------------


def _run_batch(
    args: argparse.Namespace,
    ensemble: FoulPredictionEnsemble,
) -> list[dict[str, Any]]:
    batch_path = Path(args.batch_file)
    if not batch_path.is_file():
        logger.error(f"Archivo batch no encontrado: {batch_path}")
        sys.exit(1)

    matches: list[dict[str, Any]] = []
    if batch_path.suffix.lower() == ".json":
        with open(batch_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("El JSON de batch debe ser una lista de partidos.")
            sys.exit(1)
        matches = data
    elif batch_path.suffix.lower() in (".yaml", ".yml"):
        # New format: matches.yaml from parsers/matches_parser
        matches = _parse_yaml_batch_file(batch_path)
    elif batch_path.suffix.lower() == ".csv":
        with open(batch_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                matches.append(
                    {
                        "local": row.get("local") or row.get("home_team"),
                        "visitante": row.get("visitante") or row.get("away_team"),
                        "fecha": row.get("fecha") or row.get("match_date"),
                        "jornada": int(row["jornada"]) if row.get("jornada") else None,
                        "arbitro": row.get("arbitro"),
                    }
                )
    else:
        logger.error("Formato no soportado. Usa .json, .yaml o .csv.")
        sys.exit(1)

    # Load overlay catalog once if --narratives is provided
    batch_catalog = None
    narratives_dir: Path | None = None
    if getattr(args, "narratives", None):
        narratives_dir = Path(args.narratives)
        try:
            from overlay.rules import load_catalog as _load_catalog

            batch_catalog = _load_catalog(ROOT / "overlay" / "rules.yaml")
        except Exception as exc:
            logger.warning(f"Could not load overlay rule catalog: {exc}")

    overlay_log_dir = Path(
        getattr(args, "overlay_log_dir", None) or (ROOT / "overlay" / "logs")
    )

    results: list[dict[str, Any]] = []
    for idx, m in enumerate(matches, 1):
        home = m.get("local") or ""
        away = m.get("visitante") or ""
        fecha = m.get("fecha") or args.fecha
        logger.info(f"[{idx}/{len(matches)}] {home} vs {away}")
        try:
            feat = generate_features(
                equipo_local=home,
                equipo_visitante=away,
                jornada=m.get("jornada") or args.jornada,
                fecha_partido=fecha,
                arbitro=m.get("arbitro") or args.arbitro,
                features_profile=args.features_profile,
                refresh_data=args.refresh_data,
            )
            # Look up per-match narrative in --narratives dir
            narrative_for_match = None
            if narratives_dir is not None and batch_catalog is not None:
                narr_path = _find_narrative_for_match(
                    narratives_dir=narratives_dir,
                    home=home,
                    away=away,
                    date=str(fecha),
                )
                if narr_path is not None:
                    try:
                        from overlay.loader import load_narrative as _load_narr

                        narrative_for_match = _load_narr(narr_path)
                        logger.info(f"  Narrative found: {narr_path.name}")
                    except Exception as exc:
                        logger.warning(f"  Could not load narrative {narr_path}: {exc}")
                else:
                    logger.info(f"  No narrative found for {home} vs {away} on {fecha}")

            results.append(
                _prediction_to_dict(
                    ensemble,
                    feat,
                    narrative=narrative_for_match,
                    catalog=batch_catalog if narrative_for_match else None,
                    overlay_log_dir=overlay_log_dir,
                )
            )
        except Exception as exc:
            logger.error(f"  Error en {home} vs {away}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _apply_narrative_p1(state: dict[str, Any], narrative_path: str) -> dict[str, Any]:
    """Load a narrative YAML and apply the P1 objective override to state.

    Returns a new (deep-copied) state dict with patched objectives.
    Exits with code 1 on any parse/validation error.
    """
    from overlay.loader import load_narrative
    from overlay.objective import inject_objectives_into_state

    try:
        narr = load_narrative(narrative_path)
    except FileNotFoundError as exc:
        logger.error(f"Narrative file not found: {exc}")
        sys.exit(1)
    except (ValueError, Exception) as exc:
        logger.error(f"Error loading narrative: {exc}")
        sys.exit(1)

    if narr.objectives is not None:
        logger.info(
            f"P1 overlay: applying objective override for "
            f"{list(narr.objectives.keys())} side(s)."
        )
        return inject_objectives_into_state(state, narr)
    return state


def _generate_features_with_state(
    equipo_local: str,
    equipo_visitante: str,
    state: dict[str, Any],
    *,
    jornada: int | None,
    fecha_partido: str | None,
    arbitro: str | None,
    features_profile: str | None,
) -> dict[str, Any]:
    """Generate features using a pre-loaded (possibly patched) state dict.

    Calls build_features directly (bypassing generate_features which re-fetches
    state from Supabase), so that any P1 objective patches are applied.
    """
    return build_features(
        state=state,
        equipo_local_input=equipo_local,
        equipo_visitante_input=equipo_visitante,
        jornada=jornada,
        arbitro_input=arbitro,
        arbitraje_source=("manual" if arbitro else "not_available"),
        fecha_partido_input=fecha_partido,
        features_profile=features_profile,
    )


# ---------------------------------------------------------------------------
# Run-dir support (Batch 3 — predecir-jornada-v2)
# ---------------------------------------------------------------------------


def _seed_rng(seed: int) -> None:
    """Seed all RNGs for deterministic behavior.

    Belt-and-suspenders: models use .eval() + torch.no_grad() so no stochastic
    ops are expected, but seeding ensures any future stochastic code paths
    remain deterministic.
    """
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
    except ImportError:
        pass


def _resolve_run_dir_paths(args: argparse.Namespace) -> None:
    """Resolve implicit paths from --run-dir.

    When --run-dir is set, the following are inferred (unless explicitly set):
      - args.overlay_log_dir = {run_dir}/prediction/overlay_logs
      - args.output_json     = {run_dir}/prediction/prediction.json
      - args.narratives      = {run_dir}/input/narratives

    Also reads manifest.yaml and seeds the RNG with manifest.seed.

    Raises FileNotFoundError if {run_dir}/manifest.yaml does not exist.
    """
    if not getattr(args, "run_dir", None):
        return

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.yaml not found in run-dir: {manifest_path}. "
            f"Did you forget to call runs.lifecycle.start_run()?"
        )

    # Read seed (avoid hard dependency on runs.manifest at module-load time)
    import yaml as _yaml

    manifest_data = _yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    seed = int(manifest_data.get("seed", 42))
    _seed_rng(seed)

    # Set implicit paths only if not already provided
    if not args.overlay_log_dir:
        args.overlay_log_dir = str(run_dir / "prediction" / "overlay_logs")
    if not args.output_json:
        args.output_json = str(run_dir / "prediction" / "prediction.json")
    if not args.narratives:
        args.narratives = str(run_dir / "input" / "narratives")


def _parse_yaml_batch_file(path: Path) -> list[dict[str, Any]]:
    """Parse a matches.yaml file (from parsers/matches_parser) into batch matches.

    Maps the matches.yaml schema (home/away/date/referee) to the legacy batch
    format (local/visitante/fecha/arbitro).

    Args:
        path: Path to matches.yaml.

    Returns:
        List of match dicts in legacy batch format.
    """
    import yaml as _yaml

    with open(path, encoding="utf-8") as f:
        data = _yaml.safe_load(f) or {}

    jornada = data.get("jornada")
    raw_matches = data.get("matches", []) or []

    result: list[dict[str, Any]] = []
    for m in raw_matches:
        result.append(
            {
                "local": m.get("home", ""),
                "visitante": m.get("away", ""),
                "fecha": m.get("date"),
                "arbitro": m.get("referee"),
                "jornada": int(jornada) if jornada is not None else None,
            }
        )
    return result


# ---------------------------------------------------------------------------


def _run_validate_narrative(path_str: str) -> None:
    """Validate a narrative YAML file and exit.

    Exits 0 on success, 1 on any error (file not found, parse error,
    schema violation). Never invokes the prediction pipeline.
    """
    from pathlib import Path as _Path

    from overlay.loader import load_narrative
    from overlay.schema import Narrative as _Narrative

    path = _Path(path_str)
    try:
        narr: _Narrative = load_narrative(path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, Exception) as exc:
        print(f"ERROR validating {path.name}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"OK — {path.name} is valid. "
        f"Match: {narr.match.home} vs {narr.match.away} | "
        f"confidence_level={narr.confidence_level}"
    )
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predice faltas para partidos de La Liga usando el ensemble entrenado."
    )
    parser.add_argument("--local", default="Real Madrid", help="Equipo local")
    parser.add_argument("--visitante", default="Ath Madrid", help="Equipo visitante")
    parser.add_argument(
        "--jornada", type=int, default=29, help="Numero de jornada (1-38)"
    )
    parser.add_argument(
        "--fecha",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Fecha YYYY-MM-DD",
    )
    parser.add_argument("--arbitro", default=None, help="Nombre del arbitro (opcional)")

    parser.add_argument(
        "--features-profile",
        dest="features_profile",
        choices=["prediction", "training", "minimal"],
        default=None,
        help="Perfil de features a usar",
    )
    parser.add_argument(
        "--refresh-data",
        dest="refresh_data",
        action="store_true",
        help="Fuerza recarga del estado desde Supabase",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        metavar="PATH",
        default=None,
        help="Guarda la prediccion en este fichero JSON",
    )
    parser.add_argument(
        "--batch-file",
        dest="batch_file",
        metavar="PATH",
        default=None,
        help="Fichero JSON o CSV con lista de partidos para prediccion en lote",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Nivel de logging",
    )
    parser.add_argument(
        "--validate-narrative",
        dest="validate_narrative",
        metavar="PATH",
        default=None,
        help=(
            "Valida un fichero YAML de narrativa sin ejecutar la prediccion. "
            "Sale con 0 si es valido, con 1 si hay errores."
        ),
    )
    parser.add_argument(
        "--narrative",
        dest="narrative",
        metavar="PATH",
        default=None,
        help=(
            "Ruta a un fichero YAML de narrativa para aplicar el overlay P1 "
            "(objectives antes de generar features) y P3/P4 post-prediccion."
        ),
    )
    parser.add_argument(
        "--narratives",
        dest="narratives",
        metavar="DIR",
        default=None,
        help=(
            "Directorio con YAMLs de narrativa para modo batch. "
            "Naming: {home}_vs_{away}_{date}.yaml. "
            "Partidos sin YAML se predicen normalmente (sin overlay)."
        ),
    )
    parser.add_argument(
        "--overlay-log-dir",
        dest="overlay_log_dir",
        metavar="DIR",
        default=None,
        help="Directorio donde escribir los logs de overlay (default: overlay/logs).",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        metavar="PATH",
        default=None,
        help=(
            "Run directory — routes all outputs into run folder structure "
            "(predecir-jornada-v2). Implicitly sets --overlay-log-dir, "
            "--output-json, --narratives. Reads seed from manifest.yaml."
        ),
    )

    args = parser.parse_args()

    # Resolve --run-dir implicit paths (predecir-jornada-v2)
    if getattr(args, "run_dir", None):
        _resolve_run_dir_paths(args)

    # --validate-narrative: parse + validate YAML only, exit before prediction
    if args.validate_narrative is not None:
        _run_validate_narrative(args.validate_narrative)
        return  # unreachable — _run_validate_narrative always sys.exit()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # -- Estado historico --
    logger.info("Cargando estado historico desde Supabase...")
    try:
        state = _load_state(refresh=args.refresh_data)
        logger.info(
            f"Estado cargado. Equipos conocidos: {len(state.get('scores', {}))}"
        )
    except Exception as exc:
        logger.error(f"No se pudo cargar el estado: {exc}")
        sys.exit(1)

    # -- Validacion (solo modo single) --
    if not args.batch_file:
        _validate_single(args, state)

    # -- Ensemble --
    logger.info(f"Cargando ensemble desde {CHECKPOINT_DIR}...")
    try:
        ensemble = _load_ensemble()
        logger.info("Ensemble cargado correctamente.")
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"No se pudo cargar el ensemble: {exc}")
        sys.exit(1)

    # -- Prediccion --
    if args.batch_file:
        results = _run_batch(args, ensemble)
        output_json = json.dumps(results, indent=2, ensure_ascii=False)
        if args.output_json:
            Path(args.output_json).write_text(output_json, encoding="utf-8")
            logger.info(f"Resultados guardados en {args.output_json}")
        else:
            print(output_json)
    else:
        logger.info(
            f"Generando features: {args.local} vs {args.visitante} "
            f"| J{args.jornada} | {args.fecha}"
        )
        try:
            # P1 objective override: if --narrative is supplied and has
            # P1 objectives: if --narrative is supplied, patch state['objectives'] BEFORE feature gen.
            feat_state = state
            if args.narrative is not None:
                feat_state = _apply_narrative_p1(state, args.narrative)

            feat = _generate_features_with_state(
                equipo_local=args.local,
                equipo_visitante=args.visitante,
                state=feat_state,
                jornada=args.jornada,
                fecha_partido=args.fecha,
                arbitro=args.arbitro,
                features_profile=args.features_profile,
            )
        except Exception as exc:
            logger.error(f"Error generando features: {exc}")
            sys.exit(1)

        # Salida legible siempre
        _print_prediction(ensemble, feat)

        # JSON opcional (with optional overlay P3/P4)
        if args.output_json or args.narrative:
            try:
                # Load overlay components if --narrative supplied
                overlay_narrative = None
                overlay_catalog = None
                overlay_log_dir = (
                    Path(args.overlay_log_dir)
                    if args.overlay_log_dir
                    else (ROOT / "overlay" / "logs")
                )

                if args.narrative is not None:
                    from overlay.loader import load_narrative as _load_narr
                    from overlay.rules import load_catalog as _load_catalog

                    try:
                        overlay_narrative = _load_narr(args.narrative)
                        overlay_catalog = _load_catalog(ROOT / "overlay" / "rules.yaml")
                    except Exception as exc:
                        logger.warning(f"Could not load overlay components: {exc}")

                result = _prediction_to_dict(
                    ensemble,
                    feat,
                    narrative=overlay_narrative,
                    catalog=overlay_catalog,
                    overlay_log_dir=overlay_log_dir,
                )

                if args.output_json:
                    Path(args.output_json).write_text(
                        json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    logger.info(f"Prediccion guardada en {args.output_json}")
            except Exception as exc:
                logger.error(f"No se pudo escribir el JSON: {exc}")
                sys.exit(1)

    logger.info("Listo.")


if __name__ == "__main__":
    main()
