"""Unit tests for PyTorch model adapter implementations.

TDD cycle:
  RED   — tests fail against NotImplementedError stubs
  GREEN — tests pass after real adapter implementations
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from HWFP.core.domain.exceptions import ModelInputError
from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState
from HWFP.serving.adapters._feature_keys import CANONICAL_FEATURE_KEYS

_N = len(CANONICAL_FEATURE_KEYS)


# ---------------------------------------------------------------------------
# Fakes — no I/O, no model weights, no legacy imports needed
# ---------------------------------------------------------------------------


class _FakeLegacyPMF:
    """Minimal stand-in for prediction_models FoulPMF (61-element probs array)."""

    def __init__(self, mean: float = 25.0) -> None:
        self.probs = np.zeros(61)
        center = max(0, min(60, int(round(mean))))
        self.probs[center] = 1.0


class _FakeMatchPrediction:
    def __init__(self, mean: float = 25.0) -> None:
        self.pmf_total = _FakeLegacyPMF(mean)
        self.match_info: dict = {}


class _FakeEnsemble:
    """Drop-in fake for FoulPredictionEnsemble. Returns deterministic output."""

    def predict(self, match_dict: dict) -> _FakeMatchPrediction:
        return _FakeMatchPrediction(mean=25.0)

    def _register_profiles_from_features(self, feature_dicts: list) -> None:
        pass


def _valid_features(value: float = 0.5) -> FeatureVector:
    return tuple(value for _ in CANONICAL_FEATURE_KEYS)


def _make_match() -> Match:
    return Match(
        match_id="m1",
        home_team_id="Real Madrid",
        away_team_id="Barcelona",
        kickoff=datetime(2025, 10, 26, 20, 0),
        referee_id="DE BURGOS BENGOETXEA",
        competition_id="laliga",
    )


def _make_team_state(team_id: str) -> TeamState:
    return TeamState(
        team_id=team_id,
        as_of=datetime(2025, 10, 26),
        avg_fouls_per_match=12.0,
        avg_fouls_conceded=10.5,
        form_window=5,
    )


# ---------------------------------------------------------------------------
# FilesystemModelRegistry
# ---------------------------------------------------------------------------


class TestFilesystemModelRegistry:
    def test_instantiation(self, tmp_path):
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        registry = FilesystemModelRegistry(checkpoints_dir=tmp_path)
        assert registry is not None

    def test_list_manifests_empty_dir_returns_empty_tuple(self, tmp_path):
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        registry = FilesystemModelRegistry(checkpoints_dir=tmp_path)
        result = registry.list_manifests()
        assert isinstance(result, tuple)
        assert len(result) == 0

    def test_list_manifests_with_checkpoints_returns_one_manifest(self):
        from pathlib import Path

        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        checkpoints = Path(__file__).parents[3] / "prediction_models" / "checkpoints" / "ensemble"
        if not checkpoints.exists():
            pytest.skip("Real checkpoints not available")

        registry = FilesystemModelRegistry(checkpoints_dir=checkpoints)
        manifests = registry.list_manifests()
        assert len(manifests) == 1
        from HWFP.core.domain.model_status import ModelStatus
        assert manifests[0].status == ModelStatus.PRODUCTION

    def test_load_production_raises_on_missing_checkpoints(self, tmp_path):
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )
        from HWFP.core.domain.exceptions import NoProductionModelError

        registry = FilesystemModelRegistry(checkpoints_dir=tmp_path)
        with pytest.raises(NoProductionModelError):
            registry.load_production()


# ---------------------------------------------------------------------------
# PyTorchFoulModel
# ---------------------------------------------------------------------------


class TestPyTorchFoulModel:
    def test_predict_returns_foul_pmf(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        result = model.predict(_valid_features())
        assert isinstance(result, FoulPMF)

    def test_predict_pmf_sums_to_one(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        result = model.predict(_valid_features())
        assert abs(sum(result.pmf) - 1.0) <= 1e-6

    def test_predict_pmf_values_in_range(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        result = model.predict(_valid_features())
        assert all(0.0 <= p <= 1.0 for p in result.pmf)

    def test_predict_bin_edges_length(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        result = model.predict(_valid_features())
        assert len(result.bin_edges) == len(result.pmf) + 1

    def test_predict_raises_model_input_error_on_wrong_length(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        with pytest.raises(ModelInputError):
            model.predict((1.0, 2.0, 3.0))

    def test_predict_raises_model_input_error_on_nan(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        bad = tuple(float("nan") if i == 0 else 0.5 for i in range(_N))
        with pytest.raises(ModelInputError):
            model.predict(bad)

    def test_predict_raises_model_input_error_on_inf(self):
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        model = PyTorchFoulModel(ensemble=_FakeEnsemble())
        bad = tuple(float("inf") if i == 5 else 0.5 for i in range(_N))
        with pytest.raises(ModelInputError):
            model.predict(bad)

    def test_different_means_produce_different_pmfs(self):
        """Ensemble with varying output should produce different domain PMFs."""
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        class _VaryingEnsemble:
            def __init__(self, mean: float) -> None:
                self._mean = mean

            def predict(self, match_dict: dict) -> _FakeMatchPrediction:
                return _FakeMatchPrediction(mean=self._mean)

            def _register_profiles_from_features(self, feature_dicts: list) -> None:
                pass

        model_low = PyTorchFoulModel(ensemble=_VaryingEnsemble(mean=15.0))
        model_high = PyTorchFoulModel(ensemble=_VaryingEnsemble(mean=35.0))
        f = _valid_features()
        pmf_low = model_low.predict(f)
        pmf_high = model_high.predict(f)
        assert pmf_low.pmf != pmf_high.pmf


# ---------------------------------------------------------------------------
# PyTorchFeatureBuilder
# ---------------------------------------------------------------------------


class TestPyTorchFeatureBuilder:
    def _builder_with_fake_build(self, return_dict: dict):
        from HWFP.serving.adapters.pytorch_feature_builder import PyTorchFeatureBuilder

        builder = PyTorchFeatureBuilder(state_provider_fn=lambda: {})
        builder._build_features = MagicMock(return_value=return_dict)
        return builder

    def test_build_returns_tuple(self):
        flat = {k: 1.0 for k in CANONICAL_FEATURE_KEYS}
        builder = self._builder_with_fake_build(flat)
        result = builder.build(_make_match(), _make_team_state("Real Madrid"), _make_team_state("Barcelona"))
        assert isinstance(result, tuple)

    def test_build_returns_all_floats(self):
        flat = {k: 1.0 for k in CANONICAL_FEATURE_KEYS}
        builder = self._builder_with_fake_build(flat)
        result = builder.build(_make_match(), _make_team_state("Real Madrid"), _make_team_state("Barcelona"))
        assert all(isinstance(v, float) for v in result)

    def test_build_returns_canonical_length(self):
        flat = {k: 1.0 for k in CANONICAL_FEATURE_KEYS}
        builder = self._builder_with_fake_build(flat)
        result = builder.build(_make_match(), _make_team_state("Real Madrid"), _make_team_state("Barcelona"))
        assert len(result) == _N

    def test_build_missing_keys_default_to_zero(self):
        """Keys absent from the flat dict become 0.0 in the FeatureVector."""
        builder = self._builder_with_fake_build({})
        result = builder.build(_make_match(), _make_team_state("Real Madrid"), _make_team_state("Barcelona"))
        assert all(v == 0.0 for v in result)

    def test_build_passes_correct_teams_to_legacy(self):
        """build() forwards home_team_id and away_team_id to build_features."""
        flat = {k: 0.0 for k in CANONICAL_FEATURE_KEYS}
        fake = MagicMock(return_value=flat)

        from HWFP.serving.adapters.pytorch_feature_builder import PyTorchFeatureBuilder

        builder = PyTorchFeatureBuilder(state_provider_fn=lambda: {"state": True})
        builder._build_features = fake

        match = _make_match()
        builder.build(match, _make_team_state("Real Madrid"), _make_team_state("Barcelona"))

        call_kwargs = fake.call_args.kwargs
        assert call_kwargs["equipo_local_input"] == "Real Madrid"
        assert call_kwargs["equipo_visitante_input"] == "Barcelona"
        assert call_kwargs["arbitro_input"] == "DE BURGOS BENGOETXEA"
