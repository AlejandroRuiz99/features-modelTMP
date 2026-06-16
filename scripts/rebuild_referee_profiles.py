#!/usr/bin/env python3
"""
rebuild_referee_profiles.py — Standalone script to rebuild referee profiles
and optionally retrain the ModeSelector MLP.

Pipeline:

  Stage 1 — Rebuild profiles with shrinkage
  ------------------------------------------
  Reads training.parquet, calls calcular_perfiles_gmm() from features_generator
  to produce RefereeGMMParams with correct is_shrunk flags. Saves profiles.pkl
  (with backup of any existing checkpoint).

  Stage 2 — Retrain ModeSelector (unless --no-mlp)
  --------------------------------------------------
  Filters training rows to referee_n_partidos >= 8 (REQ-8).
  Computes percentile targets from weight_strict values (D2).
  Trains MLP with seed=42 (reproducible, REQ-5).

  Stage 3 — Gate evaluation
  -------------------------
  Range gate   (REQ-5):  max-min ≥ 0.20 over 10k random context vectors.
  Discriminability gate (REQ-6): 4 synthetic profiles spread ≥ 0.10.
  NLL gate     (REQ-10): MLP_NLL ≤ static_NLL + 0.01 on tune set.

  If ALL gates pass → save mode_selector.pt.
  If ANY gate fails → set use_static_fallback=True, log WARNING, skip MLP save.

Backup: profiles.pkl.bak.{YYYYMMDD_HHMMSS} is created before any overwrite.

Usage:
  python scripts/rebuild_referee_profiles.py
  python scripts/rebuild_referee_profiles.py --training-parquet prediction_models/data/training.parquet \\
      --output-dir prediction_models/checkpoints/ensemble/referee --seed 42
  python scripts/rebuild_referee_profiles.py --no-mlp   # Stage 1 only
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — allow running from any cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "prediction_models"))
sys.path.insert(0, str(_ROOT / "features_generator"))

# pylint: disable=wrong-import-position
from features_generator.transformation.referee_gmm import (  # noqa: E402
    calcular_perfiles_gmm,
)
from prediction_models.src.models.referee_gmm import (  # noqa: E402
    MIN_MATCHES_STANDALONE,
    ModeSelector,
    RefereeProfile,
    RefereeProfiler,
    compute_percentile_target,
    filter_training_rows_by_n,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAINING_PARQUET = _ROOT / "prediction_models" / "data" / "training.parquet"
DEFAULT_OUTPUT_DIR = (
    _ROOT / "prediction_models" / "checkpoints" / "ensemble" / "referee"
)
DEFAULT_SEED = 42

# Gate thresholds (from REQ-5, REQ-6, REQ-10)
RANGE_GATE_MIN = 0.20
DISCRIMINABILITY_GATE_MIN = 0.10
NLL_GATE_MAX_DELTA = 0.01  # MLP_NLL ≤ static_NLL + 0.01 to KEEP MLP
NLL_GATE_WARN_DELTA = -0.04  # MLP_NLL ≤ static_NLL - 0.04 for clean pass

# Season splits (match train.py convention)
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TUNE_SEASON = "2024-25"
TEST_SEASON = "2025-26"


# ---------------------------------------------------------------------------
# Stage 1 — Rebuild profiles with shrinkage
# ---------------------------------------------------------------------------


def _parquet_row_to_match(row: pd.Series) -> dict:
    """Convert a parquet row to the minimal match dict expected by calcular_perfiles_gmm."""
    return {
        "referee": row["referee"],
        "date": str(row.get("date", "2020-01-01")),
        "home": {"fouls": float(row.get("fouls_home", 0.0))},
        "away": {"fouls": float(row.get("fouls_away", 0.0))},
    }


def rebuild_profiles(
    df: pd.DataFrame,
    profiler: RefereeProfiler,
) -> dict:
    """Stage 1: call calcular_perfiles_gmm and register profiles.

    Args:
        df: Full training DataFrame.
        profiler: RefereeProfiler to populate.

    Returns:
        dict[referee_name, RefereeGMMParams] as returned by calcular_perfiles_gmm.
    """
    matches = [_parquet_row_to_match(row) for _, row in df.iterrows()]
    logger.info("Running calcular_perfiles_gmm on %d matches…", len(matches))
    gmm_params = calcular_perfiles_gmm(matches)
    logger.info("GMM profiles computed for %d referees.", len(gmm_params))

    shrunk_count = sum(1 for p in gmm_params.values() if p.is_shrunk)
    standalone_count = len(gmm_params) - shrunk_count
    logger.info(
        "  is_shrunk=True:  %d referees | is_shrunk=False: %d referees",
        shrunk_count,
        standalone_count,
    )

    for name, params in gmm_params.items():
        profile = RefereeProfile.from_contract(name, params.to_dict())
        profiler.register_profile(profile)
        logger.debug(
            "  Registered %s — is_shrunk=%s n=%d",
            name,
            profile.is_shrunk,
            profile.n_matches,
        )

    return gmm_params


# ---------------------------------------------------------------------------
# Stage 2 — Retrain ModeSelector
# ---------------------------------------------------------------------------


def _build_context_row(row: pd.Series, profiler: RefereeProfiler) -> list[float]:
    """Build a 9-feature context vector from a parquet row."""
    referee = row["referee"]
    profile = profiler.get_profile(referee)

    rank_diff = float(row.get("rank_diff_norm", 0.0)) * 19.0
    is_derby = bool(row.get("is_derby", False))
    season_phase = float(row.get("season_phase", 0.5))
    home_rank = float(row.get("home_rank_curr", 10.0))
    home_is_top = home_rank <= 6.0
    aggressiveness = float(row.get("aggressiveness_norm_total", 0.5))
    urgency_home = float(row.get("urgency_home", 0.5))
    urgency_away = float(row.get("urgency_away", 0.5))
    urgency_avg = 0.5 * (urgency_home + urgency_away)
    ref_pair_delta_sum = float(row.get("ref_pair_delta_sum", 0.0))
    pace_index_curr = float(row.get("pace_index_curr", 31.0))

    # Normalize per build_context_vector logic
    def _norm_ref_pair_delta_sum(x: float) -> float:
        c = max(-8.0, min(8.0, x))
        return (c / 8.0 + 1.0) * 0.5

    def _norm_pace_index(pace: float) -> float:
        return max(0.0, min(1.0, (pace - 18.0) / 24.0))

    return [
        float(is_derby),
        float(rank_diff) / 19.0,
        float(season_phase),
        float(home_is_top),
        float(profile.weight_strict),
        max(0.0, min(1.0, aggressiveness)),
        max(0.0, min(1.0, urgency_avg)),
        _norm_ref_pair_delta_sum(ref_pair_delta_sum),
        _norm_pace_index(pace_index_curr),
    ]


def _compute_targets(df: pd.DataFrame, profiler: RefereeProfiler) -> np.ndarray:
    """Compute percentile targets for ModeSelector training.

    Each row's target = percentile rank of that referee's weight_strict
    among ALL training referees (D2).

    This is computed once per unique referee and broadcast to all rows.
    """
    # Build weight_strict lookup per referee
    referee_ws = {}
    for referee in df["referee"].unique():
        referee_ws[referee] = profiler.get_profile(referee).weight_strict

    # Compute global percentile across all unique referees
    unique_refs = list(referee_ws.keys())
    ws_array = np.array([referee_ws[r] for r in unique_refs])
    percentiles = compute_percentile_target(ws_array)
    ref_to_percentile = {r: float(p) for r, p in zip(unique_refs, percentiles)}

    # Broadcast to rows
    targets = np.array([ref_to_percentile[row["referee"]] for _, row in df.iterrows()])
    return targets


def retrain_mode_selector(
    df_train_raw: pd.DataFrame,
    df_tune: pd.DataFrame,
    profiler: RefereeProfiler,
    seed: int = DEFAULT_SEED,
    epochs: int = 100,
    lr: float = 0.001,
) -> tuple[ModeSelector, bool]:
    """Stage 2 + Stage 3: filter, train, evaluate gates.

    Args:
        df_train_raw: Raw training DataFrame (pre-filter).
        df_tune: Tune set for NLL gate evaluation.
        profiler: RefereeProfiler with registered profiles.
        seed: Random seed for reproducibility.
        epochs: Training epochs.
        lr: Adam learning rate.

    Returns:
        Tuple of (trained ModeSelector or None, gate_passed: bool).
        If gate_passed=False, use_static_fallback should be set on the profiler.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Filter training rows: n >= MIN_MATCHES_STANDALONE ──────────────────
    df_train = filter_training_rows_by_n(df_train_raw)
    n_excluded = len(df_train_raw) - len(df_train)
    logger.info(
        "Training filter: %d rows (excluded %d rows with n < %d)",
        len(df_train),
        n_excluded,
        MIN_MATCHES_STANDALONE,
    )

    # REQ-9: < 30 unique referees after filtering → lower threshold + warn
    unique_referees_train = df_train["referee"].nunique()
    if unique_referees_train < 30:
        logger.warning(
            "Only %d unique referees at n≥%d threshold. Lowering to n≥5 with proportional weights.",
            unique_referees_train,
            MIN_MATCHES_STANDALONE,
        )
        df_train = filter_training_rows_by_n(df_train_raw, min_n=5)
        logger.info("Relaxed filter: %d rows retained.", len(df_train))

    # ── Build context vectors ───────────────────────────────────────────────
    contexts = np.array(
        [_build_context_row(row, profiler) for _, row in df_train.iterrows()]
    )
    targets = _compute_targets(df_train, profiler)
    logger.info(
        "Context matrix: %s | target mean=%.3f std=%.3f",
        contexts.shape,
        targets.mean(),
        targets.std(),
    )

    # ── Train MLP ──────────────────────────────────────────────────────────
    model = ModeSelector(hidden_dims=[24, 12], dropout=0.1)
    X = torch.tensor(contexts, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        pred = model(X)
        loss = nn.functional.binary_cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            logger.debug("  epoch %3d — loss=%.4f", epoch + 1, loss.item())

    model.eval()
    logger.info("ModeSelector training complete.")

    # ── Stage 3: Gate evaluation ────────────────────────────────────────────
    gate_passed = _evaluate_gates(model, profiler, df_tune)
    return model, gate_passed


# ---------------------------------------------------------------------------
# Stage 3 — Gate evaluation
# ---------------------------------------------------------------------------


def _evaluate_gates(
    model: ModeSelector,
    profiler: RefereeProfiler,
    df_tune: pd.DataFrame,
) -> bool:
    """Evaluate the three gates. Returns True if ALL pass."""
    rng = np.random.default_rng(DEFAULT_SEED)

    # ── Range gate (REQ-5) ────────────────────────────────────────────────
    X_random = torch.tensor(
        rng.uniform(0.0, 1.0, size=(10_000, ModeSelector.N_CONTEXT_FEATURES)),
        dtype=torch.float32,
    )
    with torch.no_grad():
        outputs = model(X_random).numpy()
    output_range = float(outputs.max() - outputs.min())
    logger.info(
        "Range gate:            max-min=%.4f (threshold ≥ %.2f)",
        output_range,
        RANGE_GATE_MIN,
    )
    if output_range < RANGE_GATE_MIN:
        logger.warning(
            "GATE FAILED — Range gate: max-min=%.4f < %.2f. Activating static fallback.",
            output_range,
            RANGE_GATE_MIN,
        )
        return False

    # ── Discriminability gate (REQ-6) ─────────────────────────────────────
    synthetic_refs = [
        ("syn_lenient", 0.10),
        ("syn_moderate_low", 0.40),
        ("syn_moderate_high", 0.60),
        ("syn_strict", 0.90),
    ]
    probs = []
    for ref_name, ws in synthetic_refs:
        syn_profile = RefereeProfile(
            name=ref_name,
            n_matches=20,
            mu=np.array([22.0, 30.0]),
            sigma=np.array([4.0, 4.0]),
            weights=np.array([1.0 - ws, ws]),
            is_shrunk=False,
        )
        profiler.register_profile(syn_profile)
        # Neutral context
        ctx = profiler.build_context_vector(
            ref_name,
            is_derby=False,
            rank_diff=0.0,
            season_phase=0.5,
            home_is_top=False,
        ).unsqueeze(0)
        with torch.no_grad():
            prob = float(model(ctx).item())
        probs.append(prob)
        # Remove synthetic profile
        del profiler.profiles[ref_name]

    discrim_spread = max(probs) - min(probs)
    logger.info(
        "Discriminability gate: max-min=%.4f (threshold ≥ %.2f) | probs=%s",
        discrim_spread,
        DISCRIMINABILITY_GATE_MIN,
        [f"{p:.3f}" for p in probs],
    )
    if discrim_spread < DISCRIMINABILITY_GATE_MIN:
        logger.warning(
            "GATE FAILED — Discriminability gate: spread=%.4f < %.2f. Activating static fallback.",
            discrim_spread,
            DISCRIMINABILITY_GATE_MIN,
        )
        return False

    # ── NLL gate (REQ-10) ─────────────────────────────────────────────────
    if len(df_tune) == 0:
        logger.warning("Tune set is empty — skipping NLL gate (assuming pass).")
        logger.info("All gates PASSED.")
        return True

    tune_contexts = np.array(
        [_build_context_row(row, profiler) for _, row in df_tune.iterrows()]
    )
    tune_targets = _compute_targets(df_tune, profiler)

    X_tune = torch.tensor(tune_contexts, dtype=torch.float32)
    with torch.no_grad():
        mlp_probs = model(X_tune).numpy()

    # Static baseline: return weight_strict for each referee
    static_probs = np.array(
        [
            profiler.get_profile(row["referee"]).weight_strict
            for _, row in df_tune.iterrows()
        ]
    )

    def _nll(probs: np.ndarray, targets: np.ndarray) -> float:
        probs_c = np.clip(probs, 1e-7, 1 - 1e-7)
        return float(
            -np.mean(targets * np.log(probs_c) + (1 - targets) * np.log(1 - probs_c))
        )

    mlp_nll = _nll(mlp_probs, tune_targets)
    static_nll = _nll(static_probs, tune_targets)
    delta = mlp_nll - static_nll

    logger.info(
        "NLL gate: MLP_NLL=%.4f | static_NLL=%.4f | delta=%.4f (threshold ≤ +%.2f to KEEP)",
        mlp_nll,
        static_nll,
        delta,
        NLL_GATE_MAX_DELTA,
    )

    if delta <= NLL_GATE_WARN_DELTA:
        # MLP wins clearly
        logger.info("NLL gate PASSED cleanly (MLP wins by ≥ 0.04).")
    elif delta <= NLL_GATE_MAX_DELTA:
        # Marginal pass — warn but keep MLP
        logger.warning(
            "NLL gate PASSED marginally (delta=%.4f between -0.04 and +0.01). MLP saved with WARNING.",
            delta,
        )
    else:
        logger.warning(
            "GATE FAILED — NLL gate: MLP_NLL=%.4f > static_NLL+0.01=%.4f. Activating static fallback.",
            mlp_nll,
            static_nll + NLL_GATE_MAX_DELTA,
        )
        return False

    logger.info("All gates PASSED.")
    return True


# ---------------------------------------------------------------------------
# Backup helper
# ---------------------------------------------------------------------------


def _backup_if_exists(path: Path) -> None:
    """Create a timestamped backup of path if it exists."""
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_name(f"{path.name}.bak.{ts}")
        path.rename(bak)
        logger.info("Backup created: %s", bak)


# ---------------------------------------------------------------------------
# Temporal split (local — avoids importing train.py)
# ---------------------------------------------------------------------------


def _temporal_split(
    df: pd.DataFrame,
    train_seasons: list[str],
    tune_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple temporal split for train and tune sets."""
    train_df = df[df["season"].isin(train_seasons)].copy()
    tune_df = df[df["season"] == tune_season].copy()
    return train_df, tune_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild referee profiles and optionally retrain ModeSelector."
    )
    parser.add_argument(
        "--training-parquet",
        type=Path,
        default=DEFAULT_TRAINING_PARQUET,
        help="Path to training.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for referee checkpoint (profiles.pkl, mode_selector.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for MLP training (default: 42)",
    )
    parser.add_argument(
        "--no-mlp",
        action="store_true",
        help="Skip Stage 2+3 (MLP retrain). Only rebuild profiles.pkl.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("rebuild_referee_profiles.py — start")
    logger.info("  training-parquet : %s", args.training_parquet)
    logger.info("  output-dir       : %s", args.output_dir)
    logger.info("  seed             : %d", args.seed)
    logger.info("  no-mlp           : %s", args.no_mlp)
    logger.info("=" * 60)

    # Validate inputs
    if not args.training_parquet.exists():
        logger.error("training-parquet not found: %s", args.training_parquet)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet
    logger.info("Loading training parquet…")
    df = pd.read_parquet(args.training_parquet)
    logger.info("Loaded %d rows x %d columns.", *df.shape)

    # ── Stage 1: Rebuild profiles ─────────────────────────────────────────
    logger.info("")
    logger.info("── Stage 1: Rebuild profiles ──────────────────────────────")
    profiler = RefereeProfiler()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuild_profiles(df, profiler)

    # Backup existing profiles.pkl before overwrite
    profiles_path = args.output_dir / "profiles.pkl"
    _backup_if_exists(profiles_path)

    if args.no_mlp:
        # Save without MLP — no fallback flag change
        profiler.save(args.output_dir)
        logger.info("profiles.pkl saved (Stage 1 only). Done.")
        return

    # ── Stage 2+3: Retrain ModeSelector ──────────────────────────────────
    logger.info("")
    logger.info("── Stage 2: Retrain ModeSelector ──────────────────────────")
    df_train, df_tune = _temporal_split(df, TRAIN_SEASONS, TUNE_SEASON)
    logger.info(
        "Temporal split: train=%d rows | tune=%d rows",
        len(df_train),
        len(df_tune),
    )

    model, gate_passed = retrain_mode_selector(
        df_train_raw=df_train,
        df_tune=df_tune,
        profiler=profiler,
        seed=args.seed,
    )

    if gate_passed:
        logger.info("")
        logger.info("── Stage 3: All gates PASSED ──────────────────────────────")
        profiler.mode_selector = model
        profiler.use_static_fallback = False

        # Backup mode_selector.pt
        ms_path = args.output_dir / "mode_selector.pt"
        _backup_if_exists(ms_path)

        profiler.save(args.output_dir)
        logger.info("Saved: profiles.pkl + mode_selector.pt (MLP active)")
    else:
        logger.warning("")
        logger.warning("── Stage 3: Gate(s) FAILED — static fallback active ───────")
        profiler.use_static_fallback = True
        # Save profiles with fallback flag — do NOT save/overwrite mode_selector.pt
        profiler.save(args.output_dir)
        logger.warning("Saved: profiles.pkl with use_static_fallback=True")
        logger.warning("mode_selector.pt was NOT updated.")

    logger.info("=" * 60)
    logger.info("rebuild_referee_profiles.py — done")
    logger.info("  Referees in checkpoint : %d", len(profiler.profiles))
    logger.info("  use_static_fallback    : %s", profiler.use_static_fallback)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
