"""Contract tests for FoulModel port (REQ-8, REQ-9)."""

from __future__ import annotations

import pytest

from HWFP.core.domain.foul_pmf import FoulPMF

_FEATURES = (0.1, 0.2, 0.3, 0.4)


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def foul_model(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_foul_model")
        return mod.FakeFoulModel()
    pytest.importorskip("HWFP.serving.adapters.pytorch_foul_model")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_predict_returns_foul_pmf(foul_model):
    result = foul_model.predict(_FEATURES)
    assert isinstance(result, FoulPMF)


def test_predict_pmf_sums_to_one(foul_model):
    result = foul_model.predict(_FEATURES)
    assert abs(sum(result.pmf) - 1.0) <= 1e-6


def test_predict_all_probs_non_negative(foul_model):
    result = foul_model.predict(_FEATURES)
    assert all(p >= 0.0 for p in result.pmf)


def test_predict_bin_edges_len_is_pmf_plus_one(foul_model):
    result = foul_model.predict(_FEATURES)
    assert len(result.bin_edges) == len(result.pmf) + 1


def test_foul_model_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.pytorch_foul_model")
    adapter = mod.PyTorchFoulModel()
    with pytest.raises(NotImplementedError):
        adapter.predict(_FEATURES)
