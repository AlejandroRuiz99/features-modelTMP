"""PyTorchFoulModel — FoulModel port adapter wrapping FoulPredictionEnsemble."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from HWFP.core.domain.exceptions import ModelInputError
from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.serving.adapters._feature_keys import CANONICAL_FEATURE_KEYS

_N = len(CANONICAL_FEATURE_KEYS)

_INT_KEYS = frozenset(
    {"matchday", "home_rank_curr", "away_rank_curr", "referee_n_partidos", "h2h_partidos"}
)
_BOOL_KEYS = frozenset({"is_derby", "referee_is_shrunk", "has_market_odds"})


def _synthetic_ref_name(d: dict) -> str:
    """Derive a stable synthetic referee name from the GMM params in the feature dict.

    Different referee profiles get distinct names, allowing the ensemble's
    profiler to cache them correctly across repeated predict() calls.
    """
    key = (
        f"{d.get('referee_mu_permisivo', 22.0):.2f}"
        f"_{d.get('referee_mu_estricto', 30.0):.2f}"
        f"_{d.get('referee_sigma_permisivo', 4.0):.2f}"
        f"_{d.get('referee_sigma_estricto', 4.0):.2f}"
        f"_{d.get('referee_peso_estricto', 0.5):.3f}"
    )
    return "ref_" + hashlib.md5(key.encode()).hexdigest()[:8]


def _convert_pmf(legacy_pmf: Any) -> FoulPMF:
    """Convert legacy FoulPMF (probs: ndarray shape (61,)) → domain FoulPMF.

    Uses each foul count as its own bin: bin_edges = (0, 1, ..., 61).
    """
    import numpy as np

    probs = np.asarray(legacy_pmf.probs, dtype=np.float64)
    total = probs.sum()
    if total > 0:
        probs = probs / total

    n = len(probs)
    return FoulPMF(
        pmf=tuple(float(p) for p in probs),
        bin_edges=tuple(range(n + 1)),
    )


class PyTorchFoulModel:
    """Wraps FoulPredictionEnsemble and implements the FoulModel port.

    Converts FeatureVector (tuple[float, ...] in canonical order) to the
    named dict that the legacy ensemble.predict() expects, calls it, and
    converts the resulting legacy FoulPMF to the domain FoulPMF.
    """

    def __init__(self, ensemble: Any) -> None:
        self._ensemble = ensemble

    def predict(self, features: FeatureVector) -> FoulPMF:
        if len(features) != _N:
            raise ModelInputError(
                f"FeatureVector must have exactly {_N} elements, got {len(features)}"
            )
        if any(not math.isfinite(v) for v in features):
            raise ModelInputError("FeatureVector contains NaN or infinite values")

        match_dict = self._features_to_dict(features)
        # Register a synthetic referee profile from the GMM params so the
        # ensemble profiler can look it up by the derived name during predict.
        self._ensemble._register_profiles_from_features([match_dict])

        try:
            match_prediction = self._ensemble.predict(match_dict)
        except Exception as exc:
            raise ModelInputError(f"Ensemble prediction failed: {exc}") from exc

        return _convert_pmf(match_prediction.pmf_total)

    @staticmethod
    def _features_to_dict(features: FeatureVector) -> dict:
        d: dict = {}
        for key, val in zip(CANONICAL_FEATURE_KEYS, features):
            if key in _INT_KEYS:
                d[key] = int(val)
            elif key in _BOOL_KEYS:
                d[key] = bool(val)
            else:
                d[key] = float(val)

        d["referee"] = _synthetic_ref_name(d)
        d["home_team"] = "home"
        d["away_team"] = "away"
        d["season"] = "2025-26"
        d["date"] = ""
        d["intensidad_esperada"] = "media"
        d["riesgo_disciplinario"] = "medio"
        return d
