"""Unit tests for PromoteModelUseCase — B7b TDD RED gate (T7b.5, T7b.6)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.application.promote_model import PromoteModelUseCase, PromotionGates
from HWFP.core.domain.exceptions import PromotionError, PromotionGateFailed
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import HoldoutMetrics, ModelManifest
from HWFP.core.domain.model_status import ModelStatus
from HWFP.serving.fakes.fake_model_registry import FakeModelRegistry


def _candidate_manifest(
    model_id: str = "test-candidate-001",
    nll: float = 0.5,
    brier: float = 0.18,
    ece: float = 0.03,
) -> ModelManifest:
    return ModelManifest(
        model_id=ModelId(model_id),
        trained_at=datetime(2026, 6, 16, 10, 0, 0),
        git_sha="abc123de",
        dataset_hash="hash-fixture-001",
        dataset_rows=3,
        metrics_holdout=HoldoutMetrics(nll=nll, brier=brier, calibration_ece=ece),
        gates_passed=(),
        status=ModelStatus.CANDIDATE,
    )


def test_promote_raises_on_gate_failure() -> None:
    """T7b.5: PromotionGateFailed when candidate metrics exceed absolute thresholds."""
    bad_manifest = _candidate_manifest(nll=3.0, brier=0.30, ece=0.10)
    registry = FakeModelRegistry()
    registry.register(bad_manifest, b"blob")

    uc = PromoteModelUseCase(registry=registry, gates=PromotionGates())
    with pytest.raises(PromotionGateFailed):
        uc.execute(bad_manifest.model_id)


def test_promote_succeeds_on_passing_gates() -> None:
    """T7b.6: Gates pass + NLL improves → production manifest updated."""
    # Production fixture: NLL=1.5, Brier=0.15
    registry = FakeModelRegistry.with_production_model()
    # Candidate: NLL=0.5 (gates pass, improves over production)
    good_manifest = _candidate_manifest(nll=0.5, brier=0.18, ece=0.03)
    registry.register(good_manifest, b"blob")

    uc = PromoteModelUseCase(registry=registry, gates=PromotionGates())
    uc.execute(good_manifest.model_id)

    prod_manifests = registry.list_manifests(status=ModelStatus.PRODUCTION)
    assert any(m.model_id == good_manifest.model_id for m in prod_manifests)


def test_promote_archives_prior_production() -> None:
    """After promotion, the previous production model is ARCHIVED."""
    registry = FakeModelRegistry.with_production_model()
    prior_prod_id = registry.list_manifests(status=ModelStatus.PRODUCTION)[0].model_id

    good_manifest = _candidate_manifest(nll=0.5, brier=0.18, ece=0.03)
    registry.register(good_manifest, b"blob")

    uc = PromoteModelUseCase(registry=registry, gates=PromotionGates())
    uc.execute(good_manifest.model_id)

    archived = registry.list_manifests(status=ModelStatus.ARCHIVED)
    assert any(m.model_id == prior_prod_id for m in archived)


def test_promote_first_ever_no_production_model() -> None:
    """First-ever promotion succeeds when no production exists (gates pass)."""
    registry = FakeModelRegistry()
    good_manifest = _candidate_manifest(nll=0.5, brier=0.18, ece=0.03)
    registry.register(good_manifest, b"blob")

    uc = PromoteModelUseCase(registry=registry, gates=PromotionGates())
    uc.execute(good_manifest.model_id)  # must not raise

    prod_manifests = registry.list_manifests(status=ModelStatus.PRODUCTION)
    assert any(m.model_id == good_manifest.model_id for m in prod_manifests)


def test_promote_raises_promotion_error_when_metrics_worse() -> None:
    """PromotionError bubbles from registry when candidate doesn't beat production."""
    registry = FakeModelRegistry()
    # Set up excellent production via direct registry call
    excellent = _candidate_manifest(
        model_id="excellent-model-001", nll=0.3, brier=0.10, ece=0.01
    )
    registry.register(excellent, b"blob-prod")
    registry.promote(excellent.model_id)  # direct call for test setup

    # Candidate passes absolute gates but does NOT beat production on either metric
    worse = _candidate_manifest(
        model_id="worse-model-0001", nll=0.8, brier=0.15, ece=0.03
    )
    registry.register(worse, b"blob-cand")

    uc = PromoteModelUseCase(registry=registry, gates=PromotionGates())
    with pytest.raises(PromotionError):
        uc.execute(worse.model_id)
