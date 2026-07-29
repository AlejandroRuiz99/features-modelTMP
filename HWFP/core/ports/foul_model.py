"""FoulModel port — predict foul probability mass function."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.foul_pmf import FoulPMF


class FoulModel(Protocol):
    """Predict a FoulPMF from a feature vector.

    Raises:
        ModelInputError: If the feature vector shape or type is invalid.
    Invariant:
        Returned FoulPMF satisfies abs(sum(pmf) - 1.0) <= 1e-6.
    """

    def predict(self, features: FeatureVector) -> FoulPMF: ...
