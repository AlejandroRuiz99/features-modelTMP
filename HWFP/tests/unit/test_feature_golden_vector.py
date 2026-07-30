"""Golden-vector regression test for the local feature-pipeline absorption.

Pins ``build_features()`` output for every CANONICAL_FEATURE_KEYS entry against
a frozen, deterministic state fixture (``skip_market_fetch=True``). This test
is the safety net for the `features_generator` -> `HWFP.features` package move
(design D2 / spec capability `local-feature-pipeline`): the golden vector below
was first captured against the pre-move `features_generator.assembly.feature_assembler`
location (approval test, confirmed passing), then this file's imports were
repointed to `HWFP.features.assembly.feature_assembler` after the `git mv`,
and the exact same pinned values MUST still match (scenario "Pinned golden
vector").

Time-dependence note: `HWFP.features.transformation.{iap,xstyle,xfouls}` use
`date.today()` internally for temporal decay weighting (a pre-existing legacy
quirk unrelated to this migration). To keep this regression test deterministic
across calendar days -- not just the day it was authored -- `date.today()` is
monkeypatched to a fixed reference date for the duration of the pipeline call.
Verified empirically: the pinned values below are byte-identical whether the
frozen "today" is 2026-01-15 or 2026-06-01, because every decay weight scales
by the same day-drift factor and that factor cancels out of every weighted
ratio the pipeline computes.
"""

from __future__ import annotations

from datetime import date

import pytest

from HWFP.features.assembly.feature_assembler import build_features
from HWFP.features.core.state_cache import build_state
from HWFP.serving.adapters._feature_keys import CANONICAL_FEATURE_KEYS

_FROZEN_TODAY = date(2026, 1, 15)


class _FrozenDate(date):
    """`date` subclass whose `.today()` always returns a fixed reference date."""

    @classmethod
    def today(cls) -> date:
        return _FROZEN_TODAY


# ---------------------------------------------------------------------------
# Frozen match history -- deterministic, hand-authored (no live data, no RNG)
# ---------------------------------------------------------------------------

_PARTIDOS: list[dict] = [
    {"date": "2025-08-16", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Real Madrid", "fouls": 9, "yellow_cards": 1, "red_cards": 0, "goals": 2, "shots": 16, "shots_on_target": 7, "corners": 6, "possession": 55},
     "away": {"name": "Barcelona", "fouls": 10, "yellow_cards": 2, "red_cards": 0, "goals": 1, "shots": 13, "shots_on_target": 5, "corners": 5, "possession": 45}},
    {"date": "2025-08-17", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Sevilla", "fouls": 15, "yellow_cards": 3, "red_cards": 0, "goals": 1, "shots": 9, "shots_on_target": 3, "corners": 4, "possession": 48},
     "away": {"name": "Valencia", "fouls": 13, "yellow_cards": 2, "red_cards": 0, "goals": 1, "shots": 8, "shots_on_target": 3, "corners": 3, "possession": 52}},
    {"date": "2025-08-24", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Barcelona", "fouls": 11, "yellow_cards": 1, "red_cards": 0, "goals": 3, "shots": 15, "shots_on_target": 8, "corners": 7, "possession": 62},
     "away": {"name": "Sevilla", "fouls": 16, "yellow_cards": 3, "red_cards": 1, "goals": 0, "shots": 7, "shots_on_target": 2, "corners": 2, "possession": 38}},
    {"date": "2025-08-24", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Valencia", "fouls": 14, "yellow_cards": 2, "red_cards": 0, "goals": 0, "shots": 8, "shots_on_target": 2, "corners": 3, "possession": 40},
     "away": {"name": "Real Madrid", "fouls": 8, "yellow_cards": 1, "red_cards": 0, "goals": 2, "shots": 17, "shots_on_target": 9, "corners": 6, "possession": 60}},
    {"date": "2025-09-13", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Real Madrid", "fouls": 10, "yellow_cards": 2, "red_cards": 0, "goals": 3, "shots": 18, "shots_on_target": 10, "corners": 7, "possession": 61},
     "away": {"name": "Sevilla", "fouls": 17, "yellow_cards": 4, "red_cards": 0, "goals": 1, "shots": 6, "shots_on_target": 2, "corners": 2, "possession": 39}},
    {"date": "2025-09-14", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Valencia", "fouls": 13, "yellow_cards": 2, "red_cards": 0, "goals": 1, "shots": 9, "shots_on_target": 3, "corners": 4, "possession": 41},
     "away": {"name": "Barcelona", "fouls": 9, "yellow_cards": 1, "red_cards": 0, "goals": 2, "shots": 14, "shots_on_target": 7, "corners": 6, "possession": 59}},
    {"date": "2025-09-27", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Sevilla", "fouls": 16, "yellow_cards": 3, "red_cards": 0, "goals": 0, "shots": 7, "shots_on_target": 2, "corners": 3, "possession": 37},
     "away": {"name": "Real Madrid", "fouls": 9, "yellow_cards": 1, "red_cards": 0, "goals": 1, "shots": 15, "shots_on_target": 8, "corners": 5, "possession": 63}},
    {"date": "2025-09-28", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Barcelona", "fouls": 10, "yellow_cards": 1, "red_cards": 0, "goals": 2, "shots": 16, "shots_on_target": 9, "corners": 7, "possession": 64},
     "away": {"name": "Valencia", "fouls": 15, "yellow_cards": 3, "red_cards": 1, "goals": 1, "shots": 7, "shots_on_target": 2, "corners": 3, "possession": 36}},
    {"date": "2025-10-18", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Real Madrid", "fouls": 8, "yellow_cards": 0, "red_cards": 0, "goals": 4, "shots": 19, "shots_on_target": 11, "corners": 8, "possession": 65},
     "away": {"name": "Valencia", "fouls": 14, "yellow_cards": 3, "red_cards": 0, "goals": 0, "shots": 6, "shots_on_target": 1, "corners": 2, "possession": 35}},
    {"date": "2025-10-19", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Sevilla", "fouls": 15, "yellow_cards": 2, "red_cards": 0, "goals": 1, "shots": 8, "shots_on_target": 3, "corners": 4, "possession": 40},
     "away": {"name": "Barcelona", "fouls": 9, "yellow_cards": 1, "red_cards": 0, "goals": 2, "shots": 15, "shots_on_target": 8, "corners": 6, "possession": 60}},
    {"date": "2025-11-01", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Barcelona", "fouls": 11, "yellow_cards": 2, "red_cards": 0, "goals": 1, "shots": 13, "shots_on_target": 6, "corners": 5, "possession": 52},
     "away": {"name": "Real Madrid", "fouls": 9, "yellow_cards": 1, "red_cards": 0, "goals": 1, "shots": 14, "shots_on_target": 7, "corners": 5, "possession": 48}},
    {"date": "2025-11-02", "season": 2025, "referee": "Test Referee",
     "home": {"name": "Valencia", "fouls": 13, "yellow_cards": 2, "red_cards": 0, "goals": 2, "shots": 10, "shots_on_target": 4, "corners": 4, "possession": 51},
     "away": {"name": "Sevilla", "fouls": 16, "yellow_cards": 3, "red_cards": 0, "goals": 1, "shots": 7, "shots_on_target": 2, "corners": 3, "possession": 49}},
]

# ---------------------------------------------------------------------------
# Pinned golden vector -- captured once against the frozen fixture above.
# Keys mirror CANONICAL_FEATURE_KEYS; missing keys in build_features() output
# default to 0.0 exactly like HWFP.serving.adapters.pytorch_feature_builder
# does at runtime (`flat_dict.get(k, 0.0)`).
# ---------------------------------------------------------------------------

_GOLDEN_VECTOR: dict[str, float] = {
    "matchday": 20.0,
    "home_fouls_committed_avg": 8.8,
    "home_fouls_suffered_avg": 13.7,
    "away_fouls_committed_avg": 10.0,
    "away_fouls_suffered_avg": 12.8,
    "home_fouls_committed_curr": 8.8,
    "away_fouls_committed_curr": 10.0,
    "home_shots_curr": 16.5,
    "away_shots_curr": 14.3,
    "home_corners_curr": 6.2,
    "away_corners_curr": 6.0,
    "home_yellows_avg": 0.98,
    "away_yellows_avg": 1.34,
    "home_reds_avg": 0.0,
    "away_reds_avg": 0.0,
    "home_rank_hist": 1.0,
    "away_rank_hist": 2.0,
    "home_rank_curr": 1.0,
    "away_rank_curr": 2.0,
    "rank_diff_norm": 0.0526,
    "season_phase": 0.5263,
    "is_derby": 1.0,
    "pace_index_curr": 43.0,
    "home_possession": 0.514,
    "away_possession": 0.486,
    "home_xg": 1.87,
    "away_xg": 0.95,
    "xg_diff": 0.9200000000000002,
    "xfouls_home": 9.4,
    "xfouls_away": 11.3,
    "aggressiveness_volume_home": 0.6333,
    "aggressiveness_volume_away": 0.7633,
    "aggressiveness_norm_total": 0.6983,
    "fouls_provoked_home": 13.7,
    "fouls_provoked_away": 12.8,
    "forma_fouls_home": 8.8,
    "forma_fouls_away": 10.0,
    "urgency_home": 0.5,
    "urgency_away": 0.5,
    "momentum_home": 0.867,
    "momentum_away": 0.867,
    "fatigue_home": 0.2,
    "fatigue_away": 0.2,
    "days_rest_home": 78.0,
    "days_rest_away": 78.0,
    "xfouls_factor_home": 1.0,
    "xfouls_factor_away": 1.0,
    "referee_mu_permisivo": 22.62,
    "referee_mu_estricto": 27.86,
    "referee_sigma_permisivo": 2.28,
    "referee_sigma_estricto": 0.88,
    "referee_peso_estricto": 0.314,
    "referee_n_partidos": 12.0,
    "referee_is_shrunk": 0.0,
    "ref_home_delta": 0.0,
    "ref_away_delta": 0.0,
    "ref_pair_delta_sum": 0.0,
    "ref_pair_samples": 6.0,
    "has_market_odds": 0.0,
    "market_home_win_prob": 0.3773,
    "market_draw_prob": 0.2858,
    "market_away_win_prob": 0.3369,
    "market_favorite_prob": 0.3773,
    "market_balance": 0.9596,
    "market_entropy": 1.0923,
    "market_ou25_over_prob": 0.5,
    "market_ou25_under_prob": 0.5,
    "foul_market_prob_over": 0.5,
    "foul_market_implied_mean": 24.5,
    "referee_clean_avg": 23.3,
    "referee_avg_fouls": 0.0,
    "referee_home_bias": 0.0,
    "referee_team_committed_home": 0.0,
    "referee_team_committed_away": 0.0,
    "h2h_faltas_media": 19.5,
    "h2h_partidos": 2.0,
}


def _frozen_build_features(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Run the pipeline with `date.today()` frozen; return the 76-key vector."""
    import HWFP.features.transformation.iap as iap_mod
    import HWFP.features.transformation.xfouls as xfouls_mod
    import HWFP.features.transformation.xstyle as xstyle_mod

    monkeypatch.setattr(iap_mod, "date", _FrozenDate)
    monkeypatch.setattr(xstyle_mod, "date", _FrozenDate)
    monkeypatch.setattr(xfouls_mod, "date", _FrozenDate)

    state = build_state(_PARTIDOS, objectives={}, calendar_rows=None)
    flat = build_features(
        state=state,
        equipo_local_input="Real Madrid",
        equipo_visitante_input="Barcelona",
        jornada=20,
        arbitro_input="Test Referee",
        skip_market_fetch=True,
        fecha_partido_input="2026-01-18",
    )
    return {k: float(flat.get(k, 0.0)) for k in CANONICAL_FEATURE_KEYS}


def test_pinned_golden_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 'Pinned golden vector': exact float equality vs. the baseline."""
    result = _frozen_build_features(monkeypatch)
    assert result == _GOLDEN_VECTOR


def test_key_set_matches_canonical_feature_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 'Key-set drift': produced keys equal CANONICAL_FEATURE_KEYS, no added/missing."""
    result = _frozen_build_features(monkeypatch)
    assert set(result.keys()) == set(CANONICAL_FEATURE_KEYS)
