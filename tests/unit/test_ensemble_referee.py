"""Unit tests for ensemble.py referee-related methods.

Covers:
  - _register_profiles_from_features: is_shrunk propagation (REQ-1, REQ-2)
  - _enrich_match: referee_low_confidence flag (REQ-12)
"""

from __future__ import annotations

import logging

import numpy as np

from src.models.ensemble import FoulPredictionEnsemble
from src.models.referee_gmm import RefereeProfile, RefereeProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profiler() -> RefereeProfiler:
    """Create a fresh RefereeProfiler with no profiles registered."""
    return RefereeProfiler()


def _make_feature_dict(
    referee: str = "TestRef",
    n_partidos: int = 10,
    **overrides,
) -> dict:
    """Build a minimal feature dict suitable for _register_profiles_from_features."""
    base = {
        "referee": referee,
        "referee_mu_permisivo": 22.0,
        "referee_mu_estricto": 30.0,
        "referee_sigma_permisivo": 4.0,
        "referee_sigma_estricto": 4.0,
        "referee_peso_estricto": 0.5,
        "referee_n_partidos": n_partidos,
    }
    base.update(overrides)
    return base


def _call_register(profiler: RefereeProfiler, feature_dicts: list[dict]) -> None:
    """Call _register_profiles_from_features via the ensemble class."""
    ensemble = FoulPredictionEnsemble.__new__(FoulPredictionEnsemble)
    ensemble.referee_profiler = profiler
    ensemble._register_profiles_from_features(feature_dicts)


# ---------------------------------------------------------------------------
# T1.1 — REQ-1: is_shrunk propagation when key IS present
# ---------------------------------------------------------------------------


class TestRegisterProfilesIshrunkPresent:
    """REQ-1: referee_is_shrunk key present → RefereeProfile.is_shrunk matches."""

    def test_is_shrunk_true_propagated(self) -> None:
        """Feature dict with referee_is_shrunk=True → profile.is_shrunk is True."""
        profiler = _make_profiler()
        feat = _make_feature_dict(
            referee="ShrunkRef",
            n_partidos=4,
            referee_is_shrunk=True,
        )
        _call_register(profiler, [feat])
        profile = profiler.get_profile("ShrunkRef")
        assert profile.is_shrunk is True, (
            f"Expected is_shrunk=True for shrunk referee, got {profile.is_shrunk}"
        )

    def test_is_shrunk_false_propagated(self) -> None:
        """Feature dict with referee_is_shrunk=False → profile.is_shrunk is False."""
        profiler = _make_profiler()
        feat = _make_feature_dict(
            referee="StandaloneRef",
            n_partidos=15,
            referee_is_shrunk=False,
        )
        _call_register(profiler, [feat])
        profile = profiler.get_profile("StandaloneRef")
        assert profile.is_shrunk is False, (
            f"Expected is_shrunk=False for standalone referee, got {profile.is_shrunk}"
        )


# ---------------------------------------------------------------------------
# T1.3 — REQ-2: Legacy dict without is_shrunk → infer from n + emit WARNING
# ---------------------------------------------------------------------------


class TestRegisterProfilesLegacyFallback:
    """REQ-2: dict missing referee_is_shrunk → infer from n_partidos + WARNING."""

    def test_legacy_n4_infers_shrunk_true(self, caplog) -> None:
        """Legacy dict, n=4 → is_shrunk=True AND WARNING emitted."""
        profiler = _make_profiler()
        feat = _make_feature_dict(referee="LegacyRef4", n_partidos=4)
        # Note: no 'referee_is_shrunk' key

        with caplog.at_level(logging.WARNING):
            _call_register(profiler, [feat])

        profile = profiler.get_profile("LegacyRef4")
        assert profile.is_shrunk is True, (
            f"n=4 without key → expected is_shrunk=True, got {profile.is_shrunk}"
        )
        assert any(
            "LegacyRef4" in r.message or "warning" in r.message.lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), (
            f"Expected a WARNING log for legacy dict. Records: {[r.message for r in caplog.records]}"
        )

    def test_legacy_n0_infers_shrunk_true(self, caplog) -> None:
        """Legacy dict, n=0 → is_shrunk=True AND WARNING emitted."""
        profiler = _make_profiler()
        feat = _make_feature_dict(referee="UnknownRef", n_partidos=0)

        with caplog.at_level(logging.WARNING):
            _call_register(profiler, [feat])

        profile = profiler.get_profile("UnknownRef")
        assert profile.is_shrunk is True, (
            f"n=0 without key → expected is_shrunk=True, got {profile.is_shrunk}"
        )
        assert len(caplog.records) > 0, "Expected at least one WARNING log"

    def test_legacy_n15_infers_shrunk_false(self, caplog) -> None:
        """Legacy dict, n=15 → is_shrunk=False AND WARNING emitted."""
        profiler = _make_profiler()
        feat = _make_feature_dict(referee="LegacyRef15", n_partidos=15)

        with caplog.at_level(logging.WARNING):
            _call_register(profiler, [feat])

        profile = profiler.get_profile("LegacyRef15")
        assert profile.is_shrunk is False, (
            f"n=15 without key → expected is_shrunk=False, got {profile.is_shrunk}"
        )
        # A warning should still be emitted (legacy path taken)
        assert len(caplog.records) > 0, (
            "Expected at least one WARNING log for legacy path"
        )


# ---------------------------------------------------------------------------
# T1.5 — REQ-12: referee_low_confidence flag in _enrich_match
# ---------------------------------------------------------------------------


def _make_enrich_profiler_ensemble(
    referee: str,
    n_matches: int,
    is_shrunk: bool,
) -> FoulPredictionEnsemble:
    """
    Create an ensemble instance with one registered profile, ready for _enrich_match.
    """
    profiler = _make_profiler()
    profile = RefereeProfile(
        name=referee,
        n_matches=n_matches,
        mu=np.array([22.0, 30.0]),
        sigma=np.array([4.0, 4.0]),
        weights=np.array([0.5, 0.5]),
        is_shrunk=is_shrunk,
    )
    profiler.register_profile(profile)

    ensemble = FoulPredictionEnsemble.__new__(FoulPredictionEnsemble)
    ensemble.referee_profiler = profiler
    return ensemble


def _minimal_match_dict(referee: str) -> dict:
    """Minimal feature dict for _enrich_match."""
    return {
        "referee": referee,
        "is_derby": False,
        "home_rank_curr": 5,
        "away_rank_curr": 10,
        "season_phase": 0.5,
        "aggressiveness_norm_total": 0.5,
        "urgency_home": 0.5,
        "urgency_away": 0.5,
        "ref_pair_delta_sum": 0.0,
        "pace_index_curr": 31.0,
        "rank_diff_norm": 0.26,
    }


class TestRefereeLowConfidenceFlag:
    """REQ-12: referee_low_confidence in the enriched match dict.

    Logic: True iff profile.is_shrunk AND profile.n_matches <= 2 (D7).
    Cases (REQ-12 explicit scenarios):
      n=0  + is_shrunk=True  → True   (Quintero González: no data, pure global)
      n=1  + is_shrunk=True  → True   (Díaz de Mera: 1 match, extremely unreliable)
      n=2  + is_shrunk=True  → True   (boundary: still low-confidence)
      n=3  + is_shrunk=True  → False  (3 matches: some signal, above threshold)
      n=15 + is_shrunk=False → False  (standalone referee: normal confidence)
    """

    def test_low_confidence_true_when_shrunk_and_n_eq_0(self) -> None:
        """REQ-12 scenario: n=0 + is_shrunk=True → referee_low_confidence is True."""
        ensemble = _make_enrich_profiler_ensemble(
            referee="Quintero", n_matches=0, is_shrunk=True
        )
        result = ensemble._enrich_match(_minimal_match_dict("Quintero"))
        assert "referee_low_confidence" in result, (
            "Expected 'referee_low_confidence' key in enriched match dict"
        )
        assert result["referee_low_confidence"] is True, (
            f"Expected True for n=0 + is_shrunk=True, got {result['referee_low_confidence']}"
        )

    def test_low_confidence_true_when_shrunk_and_n_eq_1(self) -> None:
        """REQ-12 scenario: n=1 + is_shrunk=True → referee_low_confidence is True."""
        ensemble = _make_enrich_profiler_ensemble(
            referee="DiazDeMera", n_matches=1, is_shrunk=True
        )
        result = ensemble._enrich_match(_minimal_match_dict("DiazDeMera"))
        assert result["referee_low_confidence"] is True, (
            f"Expected True for n=1 + is_shrunk=True, got {result['referee_low_confidence']}"
        )

    def test_low_confidence_true_when_shrunk_and_n_eq_2(self) -> None:
        """REQ-12 scenario: n=2 + is_shrunk=True → referee_low_confidence is True (boundary)."""
        ensemble = _make_enrich_profiler_ensemble(
            referee="BoundaryRef", n_matches=2, is_shrunk=True
        )
        result = ensemble._enrich_match(_minimal_match_dict("BoundaryRef"))
        assert result["referee_low_confidence"] is True, (
            f"Expected True for n=2 + is_shrunk=True, got {result['referee_low_confidence']}"
        )

    def test_low_confidence_false_when_shrunk_and_n_eq_3(self) -> None:
        """REQ-12 scenario: n=3 + is_shrunk=True → referee_low_confidence is False (has some signal)."""
        ensemble = _make_enrich_profiler_ensemble(
            referee="SomeSignalRef", n_matches=3, is_shrunk=True
        )
        result = ensemble._enrich_match(_minimal_match_dict("SomeSignalRef"))
        assert result["referee_low_confidence"] is False, (
            f"Expected False for n=3 + is_shrunk=True, got {result['referee_low_confidence']}"
        )

    def test_low_confidence_false_when_standalone(self) -> None:
        """REQ-12 scenario: n=15 + is_shrunk=False → referee_low_confidence is False."""
        ensemble = _make_enrich_profiler_ensemble(
            referee="Lahoz", n_matches=15, is_shrunk=False
        )
        result = ensemble._enrich_match(_minimal_match_dict("Lahoz"))
        assert "referee_low_confidence" in result, (
            "Expected 'referee_low_confidence' key in enriched match dict"
        )
        assert result["referee_low_confidence"] is False, (
            f"Expected False for n=15 + is_shrunk=False, got {result['referee_low_confidence']}"
        )

    # ── Original tests renamed to preserve history ───────────────────────────
    # NOTE: test_low_confidence_true_when_shrunk_and_n_le_2 is now replaced by
    # the explicit n=0/1/2/3/15 cases above, which fully cover REQ-12.
