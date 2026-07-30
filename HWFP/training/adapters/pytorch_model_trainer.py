"""Adapter — PyTorchModelTrainer (port: ModelTrainer).

Ports scripts/train.py's training pipeline (temporal split, team averages,
FoulPredictionEnsemble.fit, post-hoc grid search, isotonic OU calibration)
behind the ModelTrainer port. Deliberately self-contained: does NOT import
scripts/train.py (that module mutates sys.path at import time and is legacy
— REQ-14 bans exactly that pattern reachable from HWFP/). The ported
functions below are the HWFP-owned, boundary-clean equivalent.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from HWFP.core.domain.feature_keys import CANONICAL_FEATURE_KEYS
from HWFP.core.domain.model_manifest import HoldoutMetrics
from HWFP.core.domain.training_example import TrainingExample

logger = logging.getLogger(__name__)

_DEFAULT_TUNE_SEASON = "2024-25"
_DEFAULT_TEST_SEASON = "2025-26"
_DEFAULT_SEED = 42
# Canonical primary market line for this project (see the "ou25" columns in
# training.parquet: market_ou25_over_prob, foul_market_prob_over, ...). Used
# to derive brier/calibration_ece, which scripts/train.py never computed
# (it only reported mae/nll) — PromotionGates.check()'s thresholds
# (brier<0.20, ece<0.05) match standard binary-classification scales for
# this line, not multiclass-PMF scales.
_DEFAULT_OU_LINE = 25.5
_DEFAULT_GRID_SEARCH_MIN_MATCHES = 30


def _match_dict(example: TrainingExample) -> dict:
    """Reconstruct the raw feature dict FoulPredictionEnsemble.fit() expects.

    `metadata` (the full raw source row, when the data source provides it)
    takes precedence over the 76-key canonical tuple, which only fills in
    defaults for keys metadata doesn't carry (e.g. a minimal fixture built
    from just the canonical tuple, with no metadata). `fouls_total` is
    always forced from `actual_fouls` — the dedicated label field is
    authoritative and is never silently overridden by a stray metadata
    value.
    """
    # Every canonical key defaults to 0.0 first (FoulPredictionEnsemble's
    # sub-components index several of them directly, e.g. NaiveBayes' fit()
    # does `m[key]`, not `m.get(key, default)` — a short/partial `features`
    # tuple, as contract tests deliberately use, must not KeyError).
    merged: dict = {key: 0.0 for key in CANONICAL_FEATURE_KEYS}
    merged.update(zip(CANONICAL_FEATURE_KEYS, example.features))
    merged.update(dict(example.metadata))
    merged["fouls_total"] = float(example.actual_fouls)
    return merged


def _derive_season(dt: datetime) -> str:
    """Football-season label ("YYYY-YY") from a kickoff date.

    Only used when an example carries no "season" in its metadata (e.g. a
    TrainingDataSource that doesn't expose it). The Spanish football
    calendar season starts around July/August.
    """
    year = dt.year
    if dt.month >= 7:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


def _season_of(example: TrainingExample) -> str:
    season = example.metadata.get("season") if example.metadata else None
    if season:
        return str(season)
    return _derive_season(example.kickoff)


def _temporal_split(
    examples: List[TrainingExample],
    tune_season: str,
    test_season: str,
) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    """3-way split ported from scripts/train.py::temporal_split + main()'s
    "train = every season before tune_season" rule (AD-6).

    Falls back to a chronological 50/25/25 split when fewer than 3 distinct
    seasons are present in the dataset (e.g. a small fixture or contract
    test with no season metadata) — the season-based split degenerates to a
    single bucket there and would starve tune/test.
    """
    seasons = {_season_of(e) for e in examples}
    if len(seasons) < 3:
        ordered = sorted(examples, key=lambda e: e.kickoff)
        n = len(ordered)
        if n < 3:
            return list(ordered), list(ordered), list(ordered)
        i_tune = max(1, int(n * 0.5))
        i_test = max(i_tune + 1, int(n * 0.75))
        return ordered[:i_tune], ordered[i_tune:i_test], ordered[i_test:]

    train = [e for e in examples if _season_of(e) < tune_season]
    tune = [e for e in examples if _season_of(e) == tune_season]
    test = [e for e in examples if _season_of(e) == test_season]
    if not train:
        train = [
            e for e in examples if _season_of(e) not in (tune_season, test_season)
        ]
    return train, tune, test


def _team_averages(
    matches: List[dict],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Ported verbatim from scripts/train.py::_team_averages."""
    committed: Dict[str, list] = {}
    suffered: Dict[str, list] = {}
    rank: Dict[str, list] = {}
    for m in matches:
        for side in ("home", "away"):
            team = m.get(f"{side}_team", "")
            if not team:
                continue
            committed.setdefault(team, []).append(
                float(m.get(f"{side}_fouls_committed_avg", 12.0))
            )
            suffered.setdefault(team, []).append(
                float(m.get(f"{side}_fouls_suffered_avg", 12.0))
            )
            rank.setdefault(team, []).append(float(m.get(f"{side}_rank_hist", 10.0)))
    return (
        {t: float(np.mean(v)) for t, v in committed.items()},
        {t: float(np.mean(v)) for t, v in suffered.items()},
        {t: float(np.mean(v)) for t, v in rank.items()},
    )


def _mae_nll(ensemble, matches: List[dict]) -> dict:
    """MAE/NLL used only for grid-search scoring — ported from
    scripts/train.py::_mae + _nll (composite selection metric)."""
    predictions, actuals = [], []
    for m in matches:
        ft = m.get("fouls_total")
        if ft is None:
            continue
        try:
            predictions.append(ensemble.predict(m))
            actuals.append(float(ft))
        except Exception:
            continue
    if not predictions:
        return {"mae": float("nan"), "nll": float("nan")}
    mae = float(
        np.mean([abs(p.expected_fouls - a) for p, a in zip(predictions, actuals)])
    )
    nlls = []
    for pred, actual in zip(predictions, actuals):
        k = max(0, min(int(actual), 60))
        p_k = float(pred.pmf_total.probs[k])
        nlls.append(-np.log(max(p_k, 1e-10)))
    return {"mae": mae, "nll": float(np.mean(nlls))}


def _grid_search(ensemble, matches: List[dict], grid: Dict[str, list]) -> dict:
    """Post-hoc hyperparameter search, ported from scripts/train.py::_grid_search
    (composite MAE/NLL selection). Grid values are configurable via
    hyperparams["grid"] so tests can use a reduced grid without changing the
    production default (identical to scripts/train.py's original values)."""
    prior_mix_values = grid.get("prior_mix", [0.10, 0.15, 0.20, 0.25, 0.35])
    min_weight_values = grid.get("min_weight", [0.01, 0.02, 0.03, 0.05])
    variance_scale_values = grid.get("variance_scale", [0.30, 0.40, 0.50, 0.60, 0.75])

    results = []
    for prior_mix in prior_mix_values:
        for min_weight in min_weight_values:
            for variance_scale in variance_scale_values:
                ensemble.weighter.prior_mix = prior_mix
                ensemble.weighter.min_weight = min_weight
                ensemble._variance_posthoc_scale = variance_scale
                metrics = _mae_nll(ensemble, matches)
                results.append(
                    {
                        "prior_mix": prior_mix,
                        "min_weight": min_weight,
                        "variance_scale": variance_scale,
                        **metrics,
                    }
                )

    if not results:
        return {}
    mean_mae = sum(r["mae"] for r in results) / len(results)
    mean_nll = sum(r["nll"] for r in results) / len(results)
    for r in results:
        mae_norm = r["mae"] / mean_mae if mean_mae else 1.0
        nll_norm = r["nll"] / mean_nll if mean_nll else 1.0
        r["composite"] = 0.6 * mae_norm + 0.4 * nll_norm
    return min(results, key=lambda r: r["composite"])


def _evaluate_holdout(ensemble, matches: List[dict], ou_line: float) -> HoldoutMetrics:
    """Holdout evaluation. `nll` matches scripts/train.py::_nll exactly.

    `brier` and `calibration_ece` are new — scripts/train.py never computed
    them (it only reported mae/nll for its own CLI summary). Both are
    computed on the canonical OU line's binary event (Over/Under `ou_line`),
    using standard formulas: Brier score and classification ECE (10
    confidence bins, weighted |mean confidence - accuracy|).
    """
    predictions, actuals = [], []
    for m in matches:
        ft = m.get("fouls_total")
        if ft is None:
            continue
        try:
            predictions.append(ensemble.predict(m))
            actuals.append(float(ft))
        except Exception:
            continue
    if not predictions:
        return HoldoutMetrics(
            nll=float("nan"), brier=float("nan"), calibration_ece=float("nan")
        )

    nlls: list[float] = []
    briers: list[float] = []
    confidences: list[float] = []
    corrects: list[float] = []
    for pred, actual in zip(predictions, actuals):
        k = max(0, min(int(actual), 60))
        p_k = float(pred.pmf_total.probs[k])
        nlls.append(-np.log(max(p_k, 1e-10)))

        p_over = float(pred.pmf_total.prob_over(ou_line))
        y_over = 1.0 if actual > ou_line else 0.0
        briers.append((p_over - y_over) ** 2)

        confidence = p_over if p_over >= 0.5 else 1.0 - p_over
        predicted_over = p_over >= 0.5
        correct = 1.0 if predicted_over == (y_over == 1.0) else 0.0
        confidences.append(confidence)
        corrects.append(correct)

    confidences_arr = np.array(confidences)
    corrects_arr = np.array(corrects)
    n_bins = 10
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences_arr)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (
            (confidences_arr >= lo) & (confidences_arr <= hi)
            if i == 0
            else (confidences_arr > lo) & (confidences_arr <= hi)
        )
        if not mask.any():
            continue
        bin_conf = float(confidences_arr[mask].mean())
        bin_acc = float(corrects_arr[mask].mean())
        ece += (mask.sum() / total) * abs(bin_conf - bin_acc)

    return HoldoutMetrics(
        nll=float(np.mean(nlls)),
        brier=float(np.mean(briers)),
        calibration_ece=float(ece),
    )


class PyTorchModelTrainer:
    """Real ModelTrainer adapter — ports scripts/train.py's pipeline.

    fit() performs: temporal split (season-based when >=3 seasons are
    present, chronological fallback otherwise) -> team averages ->
    FoulPredictionEnsemble.fit() -> optional post-hoc grid search on the
    tune split -> isotonic OU calibration -> holdout evaluation on the test
    split -> checkpoint zip with a self-describing config.json (Batch 4's
    checkpoint-authoritative load contract: load() rebuilds architecture
    from config.json before loading weights, so a freshly trained
    checkpoint never depends on model_config.yaml's — possibly different,
    future — defaults).

    hyperparams (all optional; every key has a train.py-equivalent default):
        model_config: dict passed to FoulPredictionEnsemble(config=...).
        tune_season / test_season: season labels for the 3-way split.
        no_grid_search: bool, skip the post-hoc grid search.
        grid_search_min_matches: minimum tune-set size to run grid search.
        grid: {prior_mix, min_weight, variance_scale} value-list overrides.
        ou_line: the Over/Under line used for brier/calibration_ece.
        seed: torch/numpy seed for reproducible training.
        fit_team_models: bool override (default: True when train set has
            >= 2 matches, matching TeamFoulRegressor's practical minimum).
    """

    def fit(
        self,
        examples: List[TrainingExample],
        hyperparams: Dict[str, Any],
    ) -> Tuple[bytes, HoldoutMetrics]:
        hyperparams = hyperparams or {}
        seed = int(hyperparams.get("seed", _DEFAULT_SEED))
        torch.manual_seed(seed)
        np.random.seed(seed)

        tune_season = hyperparams.get("tune_season", _DEFAULT_TUNE_SEASON)
        test_season = hyperparams.get("test_season", _DEFAULT_TEST_SEASON)
        train_examples, tune_examples, test_examples = _temporal_split(
            examples, tune_season, test_season
        )
        train_matches = [_match_dict(e) for e in train_examples]
        tune_matches = [_match_dict(e) for e in tune_examples] or train_matches
        test_matches = [_match_dict(e) for e in test_examples] or tune_matches

        team_committed, team_suffered, team_rank = _team_averages(train_matches)

        model_config: dict = dict(hyperparams.get("model_config", {}))

        from HWFP.models.ensemble import FoulPredictionEnsemble

        ensemble = FoulPredictionEnsemble(model_config)

        default_fit_team_models = len(train_matches) >= 2
        fit_team_models = bool(
            hyperparams.get("fit_team_models", default_fit_team_models)
        )
        gating_matches = tune_matches if tune_examples else None

        ensemble.fit(
            train_matches,
            team_avg_committed=team_committed,
            team_avg_suffered=team_suffered,
            team_avg_rank=team_rank,
            fit_team_models=fit_team_models,
            gating_matches=gating_matches,
        )

        grid_search_min = int(
            hyperparams.get("grid_search_min_matches", _DEFAULT_GRID_SEARCH_MIN_MATCHES)
        )
        best_params: dict = {}
        if not hyperparams.get("no_grid_search", False) and (
            len(tune_matches) >= grid_search_min
        ):
            best_params = _grid_search(ensemble, tune_matches, hyperparams.get("grid", {}))
            if best_params:
                ensemble.weighter.prior_mix = best_params["prior_mix"]
                ensemble.weighter.min_weight = best_params["min_weight"]
                ensemble._variance_posthoc_scale = best_params["variance_scale"]

        ensemble.calibrate_fit(tune_matches)

        ou_line = float(hyperparams.get("ou_line", _DEFAULT_OU_LINE))
        metrics = _evaluate_holdout(ensemble, test_matches, ou_line)

        blob = self._save_to_blob(ensemble, model_config, best_params)
        return blob, metrics

    @staticmethod
    def _save_to_blob(ensemble, model_config: dict, best_params: dict) -> bytes:
        """Save the trained ensemble to a temp dir, write a self-describing
        config.json, and zip it into an opaque blob for ModelRegistry.register().

        config.json must reflect the FINAL resolved architecture/hyperparams,
        not just the config passed at construction: `gating_network.prior_mix`
        round-trips via gating.pt's own save/load regardless, but
        `gating_network.min_weight` has no other persistence path (it is a
        plain attribute, not part of the gating network's torch state_dict)
        — grid search's chosen value would silently revert to the
        model_config.yaml default on reload without this.
        """
        import tempfile

        effective_config = dict(model_config)
        if best_params:
            gate_cfg = dict(effective_config.get("gating_network", {}))
            gate_cfg["prior_mix"] = best_params["prior_mix"]
            gate_cfg["min_weight"] = best_params["min_weight"]
            effective_config["gating_network"] = gate_cfg

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ensemble.save(tmp_dir)
            (tmp_dir / "config.json").write_text(json.dumps(effective_config))
            if best_params:
                (tmp_dir / "tuning_magic_constants.json").write_text(
                    json.dumps(best_params)
                )

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(tmp_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(tmp_dir))
            return buffer.getvalue()
