"""FakeModelTrainer — returns a fixed deterministic model blob + metrics. Zero I/O."""

from __future__ import annotations

from typing import Any

from HWFP.core.domain.model_manifest import HoldoutMetrics

_BLOB = b"fake-model-blob"
_METRICS = HoldoutMetrics(nll=0.5, brier=0.18, calibration_ece=0.03)


class FakeModelTrainer:
    """Returns a fixed model blob and HoldoutMetrics regardless of input. Zero I/O.

    fit() always returns (b"fake-model-blob", HoldoutMetrics(nll=0.5, brier=0.18, ece=0.03)).
    """

    def fit(
        self,
        examples: list[Any],
        hyperparams: dict[str, Any],
    ) -> tuple[bytes, HoldoutMetrics]:
        return _BLOB, _METRICS
