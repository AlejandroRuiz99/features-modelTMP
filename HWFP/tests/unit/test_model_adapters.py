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
_REQUIRED_CHECKPOINT_FILES = ("gating.pt", "anfis.pt", "regression.pt", "bayes.npz")


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
# HWFP.models.paths — package-relative checkpoint resolution (PR1)
# ---------------------------------------------------------------------------


class TestDefaultCheckpointsDir:
    def test_resolves_package_relative_path(self):
        from pathlib import Path

        import HWFP.models as models_pkg
        from HWFP.models.paths import default_checkpoints_dir

        expected = Path(models_pkg.__file__).resolve().parent / "checkpoints" / "ensemble"
        assert default_checkpoints_dir() == expected

    def test_does_not_reference_legacy_prediction_models_path(self):
        from HWFP.models.paths import default_checkpoints_dir

        result = str(default_checkpoints_dir())
        assert "prediction_models" not in result


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
        from HWFP.models.paths import default_checkpoints_dir
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        checkpoints = default_checkpoints_dir()
        if not checkpoints.exists():
            pytest.skip("Real checkpoints not available")

        registry = FilesystemModelRegistry(checkpoints_dir=checkpoints)
        manifests = registry.list_manifests()
        assert len(manifests) == 1
        from HWFP.core.domain.model_status import ModelStatus
        assert manifests[0].status == ModelStatus.PRODUCTION

    def test_load_production_imports_from_hwfp_models_not_dead_src_path(
        self, tmp_path, monkeypatch
    ):
        """RED (task 3.2): load_production() must not import the dead
        `src.models.ensemble` path — it must import `HWFP.models.ensemble`
        directly. `FoulPredictionEnsemble.load()` itself is monkeypatched
        to a no-op: verifying real checkpoint weight-loading is out of
        scope for this adapter-rewiring batch (see Issues Found in
        apply-progress for the pre-existing hidden_dims config drift
        this surfaced).
        """
        from HWFP.models.ensemble import FoulPredictionEnsemble
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        monkeypatch.setattr(FoulPredictionEnsemble, "load", lambda self, d: None)

        checkpoints = tmp_path / "ensemble"
        checkpoints.mkdir()
        for name in _REQUIRED_CHECKPOINT_FILES:
            (checkpoints / name).write_bytes(b"fake")

        registry = FilesystemModelRegistry(checkpoints_dir=checkpoints)
        model = registry.load_production()
        assert isinstance(model, PyTorchFoulModel)

    def test_load_production_raises_on_missing_checkpoints(self, tmp_path):
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )
        from HWFP.core.domain.exceptions import NoProductionModelError

        registry = FilesystemModelRegistry(checkpoints_dir=tmp_path)
        with pytest.raises(NoProductionModelError):
            registry.load_production()

    def test_register_writes_candidate_blob_and_manifest_without_touching_production(
        self, tmp_path
    ):
        """RED (task 3.2): register() must unzip the blob into
        {checkpoints_root}/candidates/{model_id}/ and write manifest.json,
        never touching the production checkpoints dir.
        """
        import io
        import json
        import zipfile

        from HWFP.core.domain.model_id import ModelId
        from HWFP.core.domain.model_manifest import HoldoutMetrics, ModelManifest
        from HWFP.core.domain.model_status import ModelStatus
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        production_dir = tmp_path / "ensemble"
        production_dir.mkdir()
        (production_dir / "gating.pt").write_bytes(b"prod-gating")

        registry = FilesystemModelRegistry(checkpoints_dir=production_dir)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("gating.pt", b"candidate-gating-bytes")
        blob = buf.getvalue()

        manifest = ModelManifest(
            model_id=ModelId("candidate-001"),
            trained_at=datetime(2026, 1, 1),
            git_sha="abc123",
            dataset_hash="hash-001",
            dataset_rows=100,
            metrics_holdout=HoldoutMetrics(nll=1.0, brier=0.1, calibration_ece=0.01),
            gates_passed=(),
            status=ModelStatus.CANDIDATE,
        )

        registry.register(manifest, blob)

        candidate_dir = production_dir.parent / "candidates" / "candidate-001"
        assert (candidate_dir / "gating.pt").read_bytes() == b"candidate-gating-bytes"

        manifest_path = candidate_dir / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["model_id"] == "candidate-001"
        assert data["git_sha"] == "abc123"
        assert data["metrics_holdout"]["nll"] == 1.0

        # Production checkpoint must remain untouched.
        assert (production_dir / "gating.pt").read_bytes() == b"prod-gating"

    def test_load_production_real_checkpoint_predicts_successfully(self):
        """RED→GREEN (maintainer decision, Batch 4): the real production
        checkpoint's own `config.json` declares `gating_network.hidden_dims:
        [48, 24]`, which diverges from `HWFP/models/config/model_config.yaml`'s
        default `[32, 16]`. Before the fix, `load_production()` crashed with a
        PyTorch `state_dict` size-mismatch (see apply-progress B3 Issue #1).
        No monkeypatching of `FoulPredictionEnsemble.load` here — this is the
        real, end-to-end load path against the real checkpoint on disk.
        """
        from HWFP.models.paths import default_checkpoints_dir
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        checkpoints = default_checkpoints_dir()
        if not checkpoints.exists():
            pytest.skip("Real checkpoints not available")

        registry = FilesystemModelRegistry(checkpoints_dir=checkpoints)
        model = registry.load_production()
        assert isinstance(model, PyTorchFoulModel)

        result = model.predict(_valid_features())
        assert isinstance(result, FoulPMF)
        assert abs(sum(result.pmf) - 1.0) <= 1e-6


# ---------------------------------------------------------------------------
# FoulPredictionEnsemble — checkpoint-authoritative architecture (Batch 4,
# maintainer decision 1: checkpoint config.json overrides model_config.yaml
# for architecture at load time)
# ---------------------------------------------------------------------------


class TestFoulPredictionEnsembleCheckpointConfig:
    def test_read_checkpoint_config_returns_json_contents_when_present(
        self, tmp_path
    ):
        import json

        from HWFP.models.ensemble import FoulPredictionEnsemble

        (tmp_path / "config.json").write_text(
            json.dumps({"gating_network": {"hidden_dims": [4, 2]}})
        )

        result = FoulPredictionEnsemble._read_checkpoint_config(tmp_path)
        assert result == {"gating_network": {"hidden_dims": [4, 2]}}

    def test_read_checkpoint_config_returns_empty_dict_when_absent(self, tmp_path):
        from HWFP.models.ensemble import FoulPredictionEnsemble

        result = FoulPredictionEnsemble._read_checkpoint_config(tmp_path)
        assert result == {}

    def test_load_rebuilds_gating_architecture_from_checkpoint_config(
        self, tmp_path
    ):
        """Triangulation: a checkpoint saved with a non-default gating
        architecture ([4, 2] hidden dims, vs. the [32, 16] default) must
        load successfully into a freshly-constructed default-config
        ensemble — proving `load()` rebuilds architecture from the
        checkpoint's `config.json` rather than trusting the config it was
        constructed with. Real `save()`/`load()`, no monkeypatching.
        """
        import json

        from HWFP.models.ensemble import FoulPredictionEnsemble

        small_config = {"gating_network": {"hidden_dims": [4, 2]}}
        source = FoulPredictionEnsemble(config=small_config)
        # Normalization stats default to None until fit() runs; fill with
        # dummy arrays so save()/load() round-trips as a real trained
        # checkpoint would (this batch tests architecture selection, not
        # training — the actual values are irrelevant here).
        source.layer_regression._feature_means = np.zeros(3)
        source.layer_regression._feature_stds = np.ones(3)
        source.layer_anfis._feature_mins = np.zeros(3)
        source.layer_anfis._feature_maxs = np.ones(3)
        source.save(tmp_path)
        (tmp_path / "config.json").write_text(json.dumps(small_config))

        target = FoulPredictionEnsemble()  # default config: hidden_dims [32, 16]
        target.load(tmp_path)

        gating_layers = [
            layer for layer in target.weighter.gating.net if hasattr(layer, "out_features")
        ]
        assert gating_layers[0].out_features == 4
        assert gating_layers[1].out_features == 2


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
