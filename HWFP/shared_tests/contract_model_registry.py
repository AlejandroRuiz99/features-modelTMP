"""Contract tests for ModelRegistry port (REQ-8, REQ-6, REQ-9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.exceptions import NoProductionModelError, PromotionError
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import HoldoutMetrics, ModelManifest
from HWFP.core.domain.model_status import ModelStatus


def _make_manifest(
    model_id_str: str, nll: float = 1.5, brier: float = 0.15
) -> ModelManifest:
    return ModelManifest(
        model_id=ModelId(model_id_str),
        trained_at=datetime(2026, 6, 16, 12, 0),
        git_sha="abc123",
        dataset_hash="hash-001",
        dataset_rows=1000,
        metrics_holdout=HoldoutMetrics(nll=nll, brier=brier, calibration_ece=0.03),
        gates_passed=(),
        status=ModelStatus.CANDIDATE,
    )


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def model_registry(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_model_registry")
        return mod.FakeModelRegistry()
    pytest.importorskip("HWFP.serving.adapters.filesystem_model_registry")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_load_production_empty_raises(model_registry):
    with pytest.raises(NoProductionModelError):
        model_registry.load_production()


def test_register_then_list_contains_entry(model_registry):
    manifest = _make_manifest("candidate-001")
    model_registry.register(manifest, b"blob")
    listed = model_registry.list_manifests()
    assert any(m.model_id.value == "candidate-001" for m in listed)


def test_promote_then_load_production_succeeds(model_registry):
    manifest = _make_manifest("candidate-002")
    model_registry.register(manifest, b"blob")
    model_registry.promote(manifest.model_id)
    model_registry.load_production()  # must not raise


def test_promote_worse_candidate_raises(model_registry):
    good = _make_manifest("good-model-001", nll=1.5, brier=0.15)
    model_registry.register(good, b"good-blob")
    model_registry.promote(good.model_id)

    bad = _make_manifest("bad-model-001", nll=2.0, brier=0.25)
    model_registry.register(bad, b"bad-blob")

    with pytest.raises(PromotionError):
        model_registry.promote(bad.model_id)


def test_model_registry_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.filesystem_model_registry")
    adapter = mod.FilesystemModelRegistry()
    with pytest.raises(NotImplementedError):
        adapter.load_production()
