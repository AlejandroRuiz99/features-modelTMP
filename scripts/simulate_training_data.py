"""Monte Carlo simulation to generate realistic La Liga training data.

Generates a Parquet file compatible with scripts/train.py by simulating
380 matches per season across multiple seasons using team profiles,
referee profiles, and a Negative Binomial foul generation process.

Usage:
    python scripts/simulate_training_data.py
    python scripts/simulate_training_data.py --seasons 7 --output prediction_models/data/training.parquet --seed 42
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = _ROOT / "prediction_models" / "data" / "training.parquet"

# La Liga teams with (aggressiveness_tier, quality_tier) — stable identities
# aggressiveness: 0=low, 1=medium, 2=high
# quality: 0=low, 1=medium, 2=high (affects rank, xG, possession)
_TEAM_BASE_PROFILES: dict[str, dict[str, float]] = {
    # Foul rates calibrated so sum(home + away) ~ 25.2 on average
    "Barcelona": {"aggr": 0.3, "quality": 2.0, "foul_rate": 10.2},
    "Real Madrid": {"aggr": 0.4, "quality": 2.0, "foul_rate": 10.7},
    "Ath Madrid": {"aggr": 1.5, "quality": 1.8, "foul_rate": 14.0},
    "Sociedad": {"aggr": 0.7, "quality": 1.6, "foul_rate": 11.2},
    "Villarreal": {"aggr": 0.8, "quality": 1.5, "foul_rate": 11.7},
    "Ath Bilbao": {"aggr": 1.3, "quality": 1.4, "foul_rate": 13.0},
    "Betis": {"aggr": 0.9, "quality": 1.3, "foul_rate": 12.2},
    "Sevilla": {"aggr": 1.1, "quality": 1.3, "foul_rate": 12.5},
    "Valencia": {"aggr": 1.0, "quality": 1.1, "foul_rate": 12.3},
    "Celta": {"aggr": 0.8, "quality": 1.0, "foul_rate": 11.9},
    "Vallecano": {"aggr": 1.2, "quality": 0.9, "foul_rate": 12.8},
    "Osasuna": {"aggr": 1.4, "quality": 0.9, "foul_rate": 13.5},
    "Mallorca": {"aggr": 1.3, "quality": 0.8, "foul_rate": 13.3},
    "Getafe": {"aggr": 1.8, "quality": 0.7, "foul_rate": 15.0},
    "Girona": {"aggr": 0.7, "quality": 1.4, "foul_rate": 11.4},
    "Las Palmas": {"aggr": 0.9, "quality": 0.8, "foul_rate": 12.1},
    "Espanol": {"aggr": 1.0, "quality": 0.7, "foul_rate": 12.6},
    "Alaves": {"aggr": 1.2, "quality": 0.6, "foul_rate": 13.1},
    "Leganes": {"aggr": 1.3, "quality": 0.6, "foul_rate": 13.3},
    "Levante": {"aggr": 0.9, "quality": 0.7, "foul_rate": 12.2},
    # Rotating promoted teams (share pool with below)
    "Almeria": {"aggr": 1.1, "quality": 0.5, "foul_rate": 12.8},
    "Granada": {"aggr": 1.0, "quality": 0.5, "foul_rate": 12.6},
    "Cadiz": {"aggr": 1.2, "quality": 0.5, "foul_rate": 13.0},
    "Valladolid": {"aggr": 1.1, "quality": 0.5, "foul_rate": 12.8},
    "Elche": {"aggr": 0.9, "quality": 0.5, "foul_rate": 12.4},
    "Oviedo": {"aggr": 1.0, "quality": 0.5, "foul_rate": 12.6},
}

# 20 teams compete each season — 6 fixed + rotation of 14 from the pool above
_STABLE_TEAMS = [
    "Barcelona",
    "Real Madrid",
    "Ath Madrid",
    "Ath Bilbao",
    "Betis",
    "Sevilla",
    "Valencia",
    "Celta",
    "Getafe",
    "Osasuna",
    "Sociedad",
    "Villarreal",
    "Mallorca",
    "Vallecano",
    "Alaves",
    "Las Palmas",
    "Girona",
    "Espanol",
    "Leganes",
    "Levante",
]

# Spanish referee pool (ASCII-safe names to avoid encoding issues in comparisons)
_REFEREE_POOL = [
    "Gil Manzano",
    "Martinez Munuera",
    "Sanchez Martinez",
    "Hernandez Hernandez",
    "De Burgos Bengoetxea",
    "Alberola Rojas",
    "Cordero Vega",
    "Busquets Ferrer",
    "Figueroa Vazquez",
    "Galech Apezteguia",
    "Garcia Verdura",
    "Gonzalez Fuertes",
    "Guzman Mansilla",
    "Hernandez Maeso",
    "Iglesias Villanueva",
    "Melero Lopez",
    "Munuera Montero",
    "Muniz Ruiz",
    "Ortiz Arias",
    "Pulido Santana",
    "Quintero Gonzalez",
    "Sesma Espinosa",
    "Soto Grado",
    "Diaz de Mera",
]

# Derby pairs (home_team, away_team) — always both directions
_DERBY_PAIRS: set[frozenset[str]] = {
    frozenset({"Barcelona", "Real Madrid"}),
    frozenset({"Barcelona", "Espanol"}),
    frozenset({"Real Madrid", "Ath Madrid"}),
    frozenset({"Ath Bilbao", "Sociedad"}),
    frozenset({"Sevilla", "Betis"}),
    frozenset({"Valencia", "Villarreal"}),
    frozenset({"Vallecano", "Getafe"}),
    frozenset({"Mallorca", "Espanol"}),
}

# Seasons to simulate (in order)
_ALL_SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


# ---------------------------------------------------------------------------
# Team profile generation
# ---------------------------------------------------------------------------


def _build_team_profiles(
    season_idx: int,
    rng: np.random.Generator,
    prev_profiles: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    """Build team profiles for one season with seasonal variation.

    Args:
        season_idx: Index into _ALL_SEASONS (0=first season).
        rng: Random number generator.
        prev_profiles: Team profiles from prior season (for continuity).

    Returns:
        Dictionary mapping team name → profile dict.
    """
    teams = list(_STABLE_TEAMS[:20])
    profiles: dict[str, dict[str, float]] = {}

    for team in teams:
        base = _TEAM_BASE_PROFILES.get(
            team, {"aggr": 1.0, "quality": 1.0, "foul_rate": 13.0}
        )

        # Season-to-season drift: small random walk
        if prev_profiles and team in prev_profiles:
            prev = prev_profiles[team]
            aggr = float(np.clip(prev["aggr"] + rng.normal(0.0, 0.15), 0.1, 2.5))
            quality = float(np.clip(prev["quality"] + rng.normal(0.0, 0.15), 0.1, 2.5))
            foul_rate = float(
                np.clip(prev["foul_rate"] + rng.normal(0.0, 0.4), 7.0, 20.0)
            )
        else:
            # First season: start near base with small noise
            aggr = float(np.clip(base["aggr"] + rng.normal(0.0, 0.1), 0.1, 2.5))
            quality = float(np.clip(base["quality"] + rng.normal(0.0, 0.1), 0.1, 2.5))
            foul_rate = float(
                np.clip(base["foul_rate"] + rng.normal(0.0, 0.3), 7.0, 20.0)
            )

        # Derive other attributes from aggr/quality
        fouls_suffered = float(
            np.clip(14.0 - quality * 0.7 + rng.normal(0, 0.3), 7.5, 22.0)
        )
        yellows_avg = float(np.clip(aggr * 1.2 + rng.normal(0.0, 0.2), 0.5, 6.5))
        reds_avg = float(np.clip(aggr * 0.06 + rng.normal(0.0, 0.02), 0.0, 1.0))
        shots_avg = float(
            np.clip(quality * 4.5 + 7.5 + rng.normal(0.0, 0.5), 5.0, 20.0)
        )
        corners_avg = float(
            np.clip(quality * 1.5 + 3.5 + rng.normal(0.0, 0.3), 1.0, 9.0)
        )
        xg_base = float(np.clip(quality * 0.6 + 0.8 + rng.normal(0.0, 0.1), 0.3, 2.9))
        possession_base = float(
            np.clip(quality * 0.08 + 0.42 + rng.normal(0.0, 0.02), 0.35, 0.65)
        )
        # Map aggr [0.1, 2.5] → aggressiveness_vol targeting mean ~0.85, range [0.41, 1]
        # Formula: aggr * 0.15 + 0.70, clamped to [0.41, 1.0]
        aggressiveness_vol = float(
            np.clip(aggr * 0.15 + 0.70 + rng.normal(0.0, 0.05), 0.41, 1.0)
        )

        profiles[team] = {
            "aggr": aggr,
            "quality": quality,
            "foul_rate": foul_rate,
            "fouls_suffered": fouls_suffered,
            "yellows_avg": yellows_avg,
            "reds_avg": reds_avg,
            "shots_avg": shots_avg,
            "corners_avg": corners_avg,
            "xg_base": xg_base,
            "possession_base": possession_base,
            "aggressiveness_vol": aggressiveness_vol,
        }

    return profiles


# ---------------------------------------------------------------------------
# Referee profile generation
# ---------------------------------------------------------------------------


def _build_referee_profiles(
    rng: np.random.Generator,
    prev_profiles: dict[str, dict[str, float]] | None,
    n_referees: int = 20,
) -> dict[str, dict[str, float]]:
    """Build referee profiles (GMM bimodal) for one season.

    Args:
        rng: Random number generator.
        prev_profiles: Referee profiles from prior season.
        n_referees: Number of active referees this season.

    Returns:
        Dictionary mapping referee name → GMM profile dict.
    """
    # Pick referees from pool
    pool = _REFEREE_POOL.copy()
    selected = pool[:n_referees]

    profiles: dict[str, dict[str, float]] = {}

    for ref in selected:
        if prev_profiles and ref in prev_profiles:
            prev = prev_profiles[ref]
            # Continuity: small drift year-over-year
            mu_perm = float(np.clip(prev["mu_perm"] + rng.normal(0.0, 0.5), 6.0, 28.0))
            mu_strict = float(
                np.clip(prev["mu_strict"] + rng.normal(0.0, 0.5), 22.0, 46.0)
            )
            sigma_perm = float(
                np.clip(prev["sigma_perm"] + rng.normal(0.0, 0.2), 0.0, 9.5)
            )
            sigma_strict = float(
                np.clip(prev["sigma_strict"] + rng.normal(0.0, 0.2), 0.0, 11.9)
            )
            peso_strict = float(
                np.clip(prev["peso_strict"] + rng.normal(0.0, 0.04), 0.02, 0.94)
            )
            # n_matches: cumulative career matches at season start (not reset)
            # Ranges up so that n_matches + matchday//2 spans [0, 55]
            n_matches = int(rng.integers(0, 36))
        else:
            # New referee: sample from prior distribution
            mu_perm = float(rng.normal(21.1, 3.7))
            mu_perm = float(np.clip(mu_perm, 6.0, 28.0))
            mu_strict = float(rng.normal(30.7, 3.6))
            mu_strict = float(np.clip(mu_strict, 22.0, 46.0))
            # Ensure strict > permissive
            if mu_strict <= mu_perm + 4.0:
                mu_strict = mu_perm + 4.0 + float(rng.exponential(2.0))
            mu_strict = float(np.clip(mu_strict, 22.0, 46.0))
            sigma_perm = float(np.clip(rng.normal(2.6, 1.2), 0.0, 9.5))
            sigma_strict = float(np.clip(rng.normal(3.1, 1.6), 0.0, 11.9))
            peso_strict = float(np.clip(rng.normal(0.47, 0.22), 0.02, 0.94))
            n_matches = int(rng.integers(0, 20))

        # Referee strictness factor: mu_strict drives more fouls
        # leniency_factor in [0, 1]: 0=permissive, 1=strict
        leniency = peso_strict  # weight of strict component
        ref_mu = mu_perm * (1 - leniency) + mu_strict * leniency

        profiles[ref] = {
            "mu_perm": mu_perm,
            "mu_strict": mu_strict,
            "sigma_perm": sigma_perm,
            "sigma_strict": sigma_strict,
            "peso_strict": peso_strict,
            "n_matches": n_matches,
            "ref_mu": ref_mu,  # expected total fouls under this referee
            "leniency": leniency,
        }

    return profiles


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------


def _generate_schedule(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Generate a full round-robin schedule for 20 teams (38 matchdays).

    Args:
        teams: List of 20 team names.

    Returns:
        List of 38 matchdays, each with 10 (home, away) pairs.
    """
    n = len(teams)
    assert n == 20, f"Expected 20 teams, got {n}"

    matchdays: list[list[tuple[str, str]]] = []

    # Use the "circle method" for round-robin scheduling
    # Fix team[0], rotate the rest
    fixed = teams[0]
    rotating = list(teams[1:])  # 19 teams

    first_half: list[list[tuple[str, str]]] = []
    for round_idx in range(n - 1):
        round_matches: list[tuple[str, str]] = []
        # First match: fixed team vs last in rotating
        home, away = fixed, rotating[-1]
        if round_idx % 2 == 0:
            round_matches.append((home, away))
        else:
            round_matches.append((away, home))
        # Remaining pairs: rotating[i] vs rotating[n-2-i]
        for i in range((n - 2) // 2):
            t1 = rotating[i]
            t2 = rotating[n - 2 - i]
            if (round_idx + i) % 2 == 0:
                round_matches.append((t1, t2))
            else:
                round_matches.append((t2, t1))
        first_half.append(round_matches)
        rotating = [rotating[-1]] + rotating[:-1]

    # Second half: reverse home/away for return legs
    second_half: list[list[tuple[str, str]]] = []
    for round_matches in first_half:
        second_half.append([(away, home) for (home, away) in round_matches])

    matchdays = first_half + second_half
    assert len(matchdays) == 38, f"Expected 38 matchdays, got {len(matchdays)}"
    assert all(len(md) == 10 for md in matchdays), (
        "Each matchday should have 10 matches"
    )
    return matchdays


# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------


def _generate_season_dates(season: str, rng: np.random.Generator) -> list[date]:
    """Generate 380 match dates for a season (10 matches per matchday, 38 matchdays).

    Args:
        season: Season string like '2023-24'.
        rng: Random number generator.

    Returns:
        List of 380 dates (one per match in order).
    """
    # Season starts in mid-August, ends in late May
    year_start = int(season[:4])
    start_date = date(year_start, 8, 15)

    # ~38 weeks = 266 days
    dates_all: list[date] = []
    current = start_date
    for matchday in range(38):
        # Each matchday spans a weekend (Fri-Sun) or midweek
        base_day = current + timedelta(days=matchday * 7)
        for match_num in range(10):
            # Matches spread over 3 days: some Friday, most Saturday/Sunday, some Monday
            day_offset = int(rng.integers(0, 4))
            match_date = base_day + timedelta(days=day_offset)
            dates_all.append(match_date)
    return dates_all


# ---------------------------------------------------------------------------
# Foul computation
# ---------------------------------------------------------------------------


def _compute_mu_fouls(
    home_profile: dict[str, float],
    away_profile: dict[str, float],
    ref_profile: dict[str, float],
    is_derby: bool,
    season_phase: float,
    rng: np.random.Generator,
) -> float:
    """Compute expected total fouls (mu) for a match.

    Uses team aggressiveness, quality differential, referee strictness,
    derby factor, and season phase.

    Args:
        home_profile: Home team's profile dict.
        away_profile: Away team's profile dict.
        ref_profile: Referee's profile dict.
        is_derby: Whether this is a rivalry match.
        season_phase: Match position in season [0, 1].
        rng: Random number generator for noise.

    Returns:
        Expected total fouls (mu).
    """
    # Base: SUM of both teams' per-team foul rates (total fouls = home + away)
    team_base = home_profile["foul_rate"] + away_profile["foul_rate"]

    # Referee effect: referees with higher ref_mu push toward more fouls
    # Center referee effect around 25.5 (league average total fouls)
    ref_effect = (ref_profile["ref_mu"] - 25.5) * 0.30

    # Derby bonus: +2.5 fouls average
    derby_bonus = 2.5 if is_derby else 0.0

    # Season phase: slightly more fouls in early season (teams less fit)
    # and end of season (urgency)
    # Modeled as a slight U-shape: peak at phase=0.1 and phase=0.85
    phase_effect = 0.8 * max(0.0, 0.2 - season_phase) + 0.6 * max(
        0.0, season_phase - 0.75
    )

    # Quality mismatch effect: larger gaps → more fouls (frustrated underdog)
    quality_gap = abs(home_profile["quality"] - away_profile["quality"])
    mismatch_effect = quality_gap * 0.5

    mu = team_base + ref_effect + derby_bonus + phase_effect + mismatch_effect

    # Clamp to realistic range
    mu = float(np.clip(mu, 14.0, 42.0))
    return mu


def _sample_fouls(mu: float, rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample actual fouls from Negative Binomial(mu, alpha=0.04).

    Args:
        mu: Expected total fouls.
        rng: Random number generator.

    Returns:
        Tuple of (fouls_total, fouls_home, fouls_away).
    """
    alpha = 0.010  # calibrated for std~5.7 at mu~25

    # NegBin parameterization: n = 1/alpha, p = 1/(1 + alpha*mu)
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * mu)

    fouls_total = int(nbinom.rvs(n=n_param, p=p_param, random_state=rng))
    fouls_total = max(8, min(55, fouls_total))

    # Split: home gets slightly fewer fouls (home advantage)
    # home_share ~ Beta(alpha=0.48, beta=0.52) centered ~0.49
    home_share = float(rng.beta(4.8, 5.2))  # mean ~0.48
    fouls_home = max(1, round(fouls_total * home_share))
    fouls_away = max(1, fouls_total - fouls_home)
    fouls_total = fouls_home + fouls_away

    return int(fouls_total), int(fouls_home), int(fouls_away)


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------


def _compute_rank(quality: float, rng: np.random.Generator) -> int:
    """Map quality [0.1, 2.5] to rank [1, 20] with noise.

    Higher quality → lower rank number (rank 1 = best).

    Args:
        quality: Team quality value.
        rng: Random number generator.

    Returns:
        Integer rank 1..20.
    """
    # Map quality (higher=better) to rank (lower=better)
    # quality 2.5 → rank ~1, quality 0.1 → rank ~20
    base_rank = 20.0 - quality * 7.5
    rank = base_rank + rng.normal(0.0, 1.5)
    return int(np.clip(round(rank), 1, 20))


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a float value to [lo, hi]."""
    return float(np.clip(value, lo, hi))


def _generate_market_features(
    home_quality: float,
    away_quality: float,
    rng: np.random.Generator,
    has_odds: bool,
) -> dict[str, float]:
    """Generate market odds features.

    Args:
        home_quality: Home team quality.
        away_quality: Away team quality.
        rng: Random number generator.
        has_odds: Whether market odds are available.

    Returns:
        Dictionary of market feature values.
    """
    if not has_odds:
        return {
            "market_home_win_prob": 0.0,
            "market_draw_prob": 0.0,
            "market_away_win_prob": 0.0,
            "market_favorite_prob": 0.0,
            "market_balance": 0.0,
            "market_entropy": 0.0,
            "market_ou25_over_prob": 0.0,
            "market_ou25_under_prob": 0.0,
        }

    # Quality difference drives win probabilities
    quality_diff = home_quality - away_quality
    home_win = _clamp(0.44 + quality_diff * 0.12 + rng.normal(0.0, 0.04), 0.06, 0.86)
    away_win = _clamp(0.30 - quality_diff * 0.10 + rng.normal(0.0, 0.04), 0.05, 0.83)
    draw = max(0.05, 1.0 - home_win - away_win)
    # Renormalize
    total = home_win + away_win + draw
    home_win /= total
    away_win /= total
    draw /= total

    favorite_prob = max(home_win, away_win, draw)
    # Market balance: how "balanced" the market is (1=even, 0=one-sided)
    market_balance = _clamp(
        1.0 - abs(home_win - away_win) + rng.normal(0.0, 0.05), 0.19, 1.0
    )
    # Entropy: -sum(p*log(p))
    probs = np.array([home_win, draw, away_win])
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-10))))

    # Over/under 2.5 goals — related to quality
    ou25_over = _clamp(0.48 + quality_diff * 0.05 + rng.normal(0.0, 0.03), 0.26, 0.74)
    ou25_under = 1.0 - ou25_over

    return {
        "market_home_win_prob": round(home_win, 4),
        "market_draw_prob": round(draw, 4),
        "market_away_win_prob": round(away_win, 4),
        "market_favorite_prob": round(favorite_prob, 4),
        "market_balance": round(market_balance, 4),
        "market_entropy": round(entropy, 4),
        "market_ou25_over_prob": round(ou25_over, 4),
        "market_ou25_under_prob": round(ou25_under, 4),
    }


# ---------------------------------------------------------------------------
# H2H tracker
# ---------------------------------------------------------------------------


class _H2HTracker:
    """Tracks head-to-head foul history between team pairs."""

    def __init__(self) -> None:
        self._history: dict[frozenset[str], list[float]] = {}

    def get(self, home: str, away: str) -> tuple[float, int]:
        """Get h2h mean fouls and number of prior meetings.

        Args:
            home: Home team name.
            away: Away team name.

        Returns:
            (mean_fouls, n_matches) — defaults if no history.
        """
        key = frozenset({home, away})
        history = self._history.get(key, [])
        if not history:
            return 25.0, 0
        return float(np.mean(history[-5:])), min(len(history), 5)

    def update(self, home: str, away: str, fouls_total: float) -> None:
        """Record a match result.

        Args:
            home: Home team name.
            away: Away team name.
            fouls_total: Actual total fouls.
        """
        key = frozenset({home, away})
        self._history.setdefault(key, []).append(fouls_total)


# ---------------------------------------------------------------------------
# Team season stats tracker
# ---------------------------------------------------------------------------


class _TeamStatsTracker:
    """Tracks rolling season statistics per team."""

    def __init__(self) -> None:
        self._fouls_committed: dict[str, list[float]] = {}
        self._fouls_suffered: dict[str, list[float]] = {}
        self._wins_draws_losses: dict[str, list[int]] = {}  # 1=win, 0=draw, -1=loss

    def reset(self) -> None:
        """Reset all stats for a new season."""
        self._fouls_committed.clear()
        self._fouls_suffered.clear()
        self._wins_draws_losses.clear()

    def get_fouls_committed_avg(self, team: str, base_rate: float) -> float:
        """Get season-average fouls committed.

        Args:
            team: Team name.
            base_rate: Fallback rate if no history.

        Returns:
            Average fouls committed this season.
        """
        data = self._fouls_committed.get(team, [])
        if not data:
            return base_rate
        return float(np.mean(data))

    def get_fouls_committed_curr(self, team: str, base_rate: float) -> float:
        """Get recent (last 5) fouls committed average.

        Args:
            team: Team name.
            base_rate: Fallback rate if no history.

        Returns:
            Recent fouls committed average.
        """
        data = self._fouls_committed.get(team, [])
        if not data:
            return base_rate
        return float(np.mean(data[-5:]))

    def get_fouls_suffered_avg(self, team: str, base_suffered: float) -> float:
        """Get season-average fouls suffered.

        Args:
            team: Team name.
            base_suffered: Fallback rate if no history.

        Returns:
            Average fouls suffered this season.
        """
        data = self._fouls_suffered.get(team, [])
        if not data:
            return base_suffered
        return float(np.mean(data))

    def get_momentum(self, team: str) -> float:
        """Compute momentum from recent results.

        Args:
            team: Team name.

        Returns:
            Momentum value in [0, 1].
        """
        wdl = self._wins_draws_losses.get(team, [])
        if not wdl:
            return 0.5
        recent = wdl[-5:]
        # Weighted: win=3pts, draw=1pt, loss=0pts, max=15 → normalize
        points = sum(3 if r == 1 else (1 if r == 0 else 0) for r in recent)
        return float(points / 15.0)

    def update(
        self,
        home: str,
        away: str,
        home_fouls: float,
        away_fouls: float,
        home_xg: float,
        away_xg: float,
    ) -> None:
        """Record match results for both teams.

        Args:
            home: Home team name.
            away: Away team name.
            home_fouls: Actual home fouls committed.
            away_fouls: Actual away fouls committed.
            home_xg: Home expected goals.
            away_xg: Away expected goals.
        """
        self._fouls_committed.setdefault(home, []).append(home_fouls)
        self._fouls_committed.setdefault(away, []).append(away_fouls)
        self._fouls_suffered.setdefault(home, []).append(away_fouls)
        self._fouls_suffered.setdefault(away, []).append(home_fouls)

        # Result based on xG (proxy)
        if home_xg > away_xg + 0.2:
            home_res, away_res = 1, -1
        elif away_xg > home_xg + 0.2:
            home_res, away_res = -1, 1
        else:
            home_res, away_res = 0, 0
        self._wins_draws_losses.setdefault(home, []).append(home_res)
        self._wins_draws_losses.setdefault(away, []).append(away_res)


# ---------------------------------------------------------------------------
# Ref interaction tracker
# ---------------------------------------------------------------------------


class _RefInteractionTracker:
    """Tracks referee × team interaction deltas."""

    def __init__(self) -> None:
        # ref_name → team_name → list of (actual - expected) deltas
        self._deltas: dict[str, dict[str, list[float]]] = {}

    def get_delta(self, ref: str, team: str) -> tuple[float, int]:
        """Get mean delta for ref-team pair.

        Args:
            ref: Referee name.
            team: Team name.

        Returns:
            (mean_delta, n_samples).
        """
        samples = self._deltas.get(ref, {}).get(team, [])
        if not samples:
            return 0.0, 0
        return float(np.mean(samples[-5:])), len(samples)

    def update(
        self, ref: str, home: str, away: str, fouls_total: float, mu: float
    ) -> None:
        """Record a ref-team interaction.

        Args:
            ref: Referee name.
            home: Home team name.
            away: Away team name.
            fouls_total: Actual total fouls.
            mu: Expected total fouls.
        """
        delta = (fouls_total - mu) / max(mu, 1.0)
        self._deltas.setdefault(ref, {}).setdefault(home, []).append(delta)
        self._deltas.setdefault(ref, {}).setdefault(away, []).append(delta)


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_season(
    season: str,
    season_idx: int,
    team_profiles: dict[str, dict[str, float]],
    ref_profiles: dict[str, dict[str, float]],
    h2h_tracker: _H2HTracker,
    ref_tracker: _RefInteractionTracker,
    rng: np.random.Generator,
) -> list[dict]:
    """Simulate all 380 matches of one La Liga season.

    Args:
        season: Season string e.g. '2023-24'.
        season_idx: Index of this season in the full list.
        team_profiles: Team profiles for this season.
        ref_profiles: Referee profiles for this season.
        h2h_tracker: Head-to-head history (updated in place).
        ref_tracker: Referee-team interaction tracker (updated in place).
        rng: Random number generator.

    Returns:
        List of row dicts (one per match).
    """
    teams = list(team_profiles.keys())
    schedule = _generate_schedule(teams)
    stats_tracker = _TeamStatsTracker()

    referee_names = list(ref_profiles.keys())
    n_refs = len(referee_names)

    # Generate dates
    dates = _generate_season_dates(season, rng)
    date_idx = 0

    rows: list[dict] = []

    for matchday_idx, matchday_matches in enumerate(schedule):
        matchday = matchday_idx + 1
        season_phase = (matchday - 1) / 37.0

        # Assign referees to matches in this matchday (no repeat)
        ref_pool_this_md = rng.choice(n_refs, size=min(10, n_refs), replace=False)

        for match_num, (home, away) in enumerate(matchday_matches):
            match_date = dates[date_idx] if date_idx < len(dates) else dates[-1]
            date_idx += 1

            # Assign referee
            ref_name = referee_names[
                int(ref_pool_this_md[match_num % len(ref_pool_this_md)])
            ]
            ref_prof = ref_profiles[ref_name]

            home_prof = team_profiles[home]
            away_prof = team_profiles[away]

            # Derby check
            is_derby = frozenset({home, away}) in _DERBY_PAIRS

            # 1. Compute expected fouls and sample
            mu = _compute_mu_fouls(
                home_prof, away_prof, ref_prof, is_derby, season_phase, rng
            )
            fouls_total, fouls_home, fouls_away = _sample_fouls(mu, rng)

            # 2. Compute team season stats (walk-forward: only past matches)
            home_fca = stats_tracker.get_fouls_committed_avg(
                home, home_prof["foul_rate"]
            )
            away_fca = stats_tracker.get_fouls_committed_avg(
                away, away_prof["foul_rate"]
            )
            home_fca_curr = stats_tracker.get_fouls_committed_curr(
                home, home_prof["foul_rate"]
            )
            away_fca_curr = stats_tracker.get_fouls_committed_curr(
                away, away_prof["foul_rate"]
            )
            home_fsa = stats_tracker.get_fouls_suffered_avg(
                home, home_prof["fouls_suffered"]
            )
            away_fsa = stats_tracker.get_fouls_suffered_avg(
                away, away_prof["fouls_suffered"]
            )

            # 3. Ranks
            home_rank_hist = _compute_rank(home_prof["quality"], rng)
            away_rank_hist = _compute_rank(away_prof["quality"], rng)
            home_rank_curr = _compute_rank(
                home_prof["quality"] + rng.normal(0, 0.1), rng
            )
            away_rank_curr = _compute_rank(
                away_prof["quality"] + rng.normal(0, 0.1), rng
            )
            rank_diff_norm = _clamp(
                abs(home_rank_hist - away_rank_hist) / 19.0, 0.0, 1.0
            )

            # 4. xG and possession
            home_xg = _clamp(home_prof["xg_base"] + rng.normal(0.0, 0.15), 0.0, 2.9)
            away_xg = _clamp(
                away_prof["xg_base"] * 0.7
                + rng.normal(0.0, 0.12),  # away slight penalty
                0.0,
                2.13,
            )
            xg_diff = _clamp(home_xg - away_xg, -1.71, 2.42)
            home_poss = _clamp(
                home_prof["possession_base"] + rng.normal(0.0, 0.02), 0.35, 0.65
            )
            away_poss = round(1.0 - home_poss, 4)

            # 5. xfouls — blended: 50% from team season avg + 50% from actual fouls
            # This creates r~0.35-0.50 correlation with actual fouls_home/fouls_away
            xfouls_home = _clamp(
                0.5 * home_fca + 0.5 * float(fouls_home) + rng.normal(0.0, 1.5),
                5.8,
                32.6,
            )
            xfouls_away = _clamp(
                0.5 * away_fca + 0.5 * float(fouls_away) + rng.normal(0.0, 1.5),
                4.6,
                28.2,
            )
            xfouls_factor_home = _clamp(1.02 + rng.normal(0.0, 0.01), 0.99, 1.06)
            xfouls_factor_away = _clamp(1.02 + rng.normal(0.0, 0.01), 0.99, 1.06)

            # 6. Aggressiveness
            aggr_vol_home = _clamp(
                home_prof["aggressiveness_vol"] + rng.normal(0.0, 0.05), 0.31, 1.0
            )
            aggr_vol_away = _clamp(
                away_prof["aggressiveness_vol"] + rng.normal(0.0, 0.05), 0.31, 1.0
            )
            aggr_norm_total = _clamp(
                (aggr_vol_home + aggr_vol_away) / 2.0 + rng.normal(0.0, 0.02), 0.49, 1.0
            )

            # 7. Form / momentum
            momentum_home = _clamp(
                stats_tracker.get_momentum(home) + rng.normal(0.0, 0.05), 0.0, 1.0
            )
            momentum_away = _clamp(
                stats_tracker.get_momentum(away) + rng.normal(0.0, 0.05), 0.0, 1.0
            )

            # 8. Context features
            # Urgency: increases in late season and for low-ranked teams
            urgency_home = _clamp(
                0.35
                + season_phase * 0.3
                + (home_rank_hist / 20.0) * 0.2
                + rng.normal(0.0, 0.04),
                0.29,
                0.86,
            )
            urgency_away = _clamp(
                0.35
                + season_phase * 0.3
                + (away_rank_hist / 20.0) * 0.2
                + rng.normal(0.0, 0.04),
                0.29,
                0.86,
            )
            # Fatigue: more in dense fixture periods
            fatigue_home = _clamp(0.30 * season_phase + rng.normal(0.0, 0.1), 0.0, 0.88)
            fatigue_away = _clamp(0.30 * season_phase + rng.normal(0.0, 0.1), 0.0, 0.88)
            days_rest_home = float(int(np.clip(rng.exponential(7.0) + 2, 2, 87)))
            days_rest_away = float(int(np.clip(rng.exponential(7.0) + 2, 2, 87)))

            # 9. Referee GMM profile
            ref_delta_home, ref_home_samples = ref_tracker.get_delta(ref_name, home)
            ref_delta_away, ref_away_samples = ref_tracker.get_delta(ref_name, away)
            ref_pair_delta = ref_delta_home + ref_delta_away
            ref_pair_samples = float(ref_home_samples + ref_away_samples)

            # referee_n_partidos: cumulative matches refereed (career in dataset)
            # Initial n_matches = 0 for new referees, accumulated in prev profile
            # Each matchday adds ~0.5 to this season's running count
            ref_n_partidos = int(np.clip(ref_prof["n_matches"] + matchday // 2, 0, 55))

            # 10. Market
            has_odds = bool(rng.random() > 0.03)  # ~97% have odds
            market_feats = _generate_market_features(
                home_prof["quality"], away_prof["quality"], rng, has_odds
            )

            # 11. H2H
            h2h_mean, h2h_n = h2h_tracker.get(home, away)

            # 12. Match profile features
            intensidad_options = ["alta", "media", "baja"]
            # Higher aggr or derby → more likely "alta"
            aggr_mean = (home_prof["aggr"] + away_prof["aggr"]) / 2.0
            if is_derby or aggr_mean > 1.4:
                intensidad_probs = [0.55, 0.30, 0.15]
            elif aggr_mean > 0.9:
                intensidad_probs = [0.35, 0.40, 0.25]
            else:
                intensidad_probs = [0.20, 0.45, 0.35]
            intensidad = str(rng.choice(intensidad_options, p=intensidad_probs))

            riesgo_options = ["alto", "medio", "bajo"]
            if is_derby or aggr_mean > 1.5:
                riesgo_probs = [0.55, 0.30, 0.15]
            elif aggr_mean > 1.0:
                riesgo_probs = [0.30, 0.45, 0.25]
            else:
                riesgo_probs = [0.15, 0.40, 0.45]
            riesgo = str(rng.choice(riesgo_options, p=riesgo_probs))

            pace_index = _clamp(
                (home_prof["shots_avg"] + away_prof["shots_avg"])
                + rng.normal(0.0, 2.0),
                15.1,
                50.7,
            )

            # 13. Yellows/reds
            home_yellows_avg = _clamp(
                home_prof["yellows_avg"] + rng.normal(0.0, 0.15), 0.0, 6.5
            )
            away_yellows_avg = _clamp(
                away_prof["yellows_avg"] + rng.normal(0.0, 0.15), 0.0, 8.0
            )
            home_reds_avg = _clamp(
                home_prof["reds_avg"] + rng.normal(0.0, 0.01), 0.0, 1.0
            )
            away_reds_avg = _clamp(
                away_prof["reds_avg"] + rng.normal(0.0, 0.01), 0.0, 1.0
            )

            # 14. Shots and corners (curr = recent 5-match average)
            home_shots_curr = _clamp(
                home_prof["shots_avg"] + rng.normal(0.0, 0.8), 5.0, 19.3
            )
            away_shots_curr = _clamp(
                away_prof["shots_avg"] + rng.normal(0.0, 0.8), 5.0, 19.5
            )
            home_corners_curr = _clamp(
                home_prof["corners_avg"] + rng.normal(0.0, 0.3), 1.0, 8.0
            )
            away_corners_curr = _clamp(
                away_prof["corners_avg"] + rng.normal(0.0, 0.3), 0.5, 9.0
            )

            row: dict = {
                # Identity
                "home_team": home,
                "away_team": away,
                "referee": ref_name,
                "matchday": matchday,
                "season": season,
                "date": match_date.strftime("%Y-%m-%d"),
                # Team season stats
                "home_fouls_committed_avg": round(home_fca, 2),
                "home_fouls_suffered_avg": round(home_fsa, 2),
                "away_fouls_committed_avg": round(away_fca, 2),
                "away_fouls_suffered_avg": round(away_fsa, 2),
                "home_fouls_committed_curr": round(home_fca_curr, 2),
                "away_fouls_committed_curr": round(away_fca_curr, 2),
                "home_shots_curr": round(home_shots_curr, 2),
                "away_shots_curr": round(away_shots_curr, 2),
                "home_corners_curr": round(home_corners_curr, 2),
                "away_corners_curr": round(away_corners_curr, 2),
                "home_yellows_avg": round(home_yellows_avg, 3),
                "away_yellows_avg": round(away_yellows_avg, 3),
                "home_reds_avg": round(home_reds_avg, 3),
                "away_reds_avg": round(away_reds_avg, 3),
                "fouls_provoked_home": round(home_fsa, 2),
                "fouls_provoked_away": round(away_fsa, 2),
                # Rankings
                "home_rank_hist": float(home_rank_hist),
                "away_rank_hist": float(away_rank_hist),
                "home_rank_curr": home_rank_curr,
                "away_rank_curr": away_rank_curr,
                "rank_diff_norm": round(rank_diff_norm, 4),
                # xG
                "home_xg": round(home_xg, 3),
                "away_xg": round(away_xg, 3),
                "xg_diff": round(xg_diff, 3),
                "home_possession": round(home_poss, 3),
                "away_possession": round(away_poss, 3),
                # xfouls
                "xfouls_home": round(xfouls_home, 2),
                "xfouls_away": round(xfouls_away, 2),
                "xfouls_factor_home": round(xfouls_factor_home, 4),
                "xfouls_factor_away": round(xfouls_factor_away, 4),
                # Aggressiveness
                "aggressiveness_volume_home": round(aggr_vol_home, 4),
                "aggressiveness_volume_away": round(aggr_vol_away, 4),
                "aggressiveness_norm_total": round(aggr_norm_total, 4),
                # Form
                "forma_fouls_home": round(home_fca_curr, 2),
                "forma_fouls_away": round(away_fca_curr, 2),
                "momentum_home": round(momentum_home, 3),
                "momentum_away": round(momentum_away, 3),
                # Context
                "season_phase": round(season_phase, 4),
                "is_derby": is_derby,
                "urgency_home": round(urgency_home, 3),
                "urgency_away": round(urgency_away, 3),
                "fatigue_home": round(fatigue_home, 3),
                "fatigue_away": round(fatigue_away, 3),
                "days_rest_home": days_rest_home,
                "days_rest_away": days_rest_away,
                # Referee GMM
                "referee_mu_permisivo": round(ref_prof["mu_perm"], 3),
                "referee_mu_estricto": round(ref_prof["mu_strict"], 3),
                "referee_sigma_permisivo": round(ref_prof["sigma_perm"], 3),
                "referee_sigma_estricto": round(ref_prof["sigma_strict"], 3),
                "referee_peso_estricto": round(ref_prof["peso_strict"], 3),
                "referee_n_partidos": ref_n_partidos,
                # Referee interaction
                "ref_home_delta": round(ref_delta_home, 4),
                "ref_away_delta": round(ref_delta_away, 4),
                "ref_pair_delta_sum": round(ref_pair_delta, 4),
                "ref_pair_samples": ref_pair_samples,
                # Market
                "has_market_odds": has_odds,
                **{k: v if has_odds else float(v) for k, v in market_feats.items()},
                "foul_market_prob_over": 0.5,
                "foul_market_implied_mean": 24.5,
                # Match profile
                "intensidad_esperada": intensidad,
                "riesgo_disciplinario": riesgo,
                "pace_index_curr": round(pace_index, 2),
                "h2h_faltas_media": round(h2h_mean, 2),
                "h2h_partidos": h2h_n,
                # Target
                "fouls_total": float(fouls_total),
                "fouls_home": float(fouls_home),
                "fouls_away": float(fouls_away),
            }

            rows.append(row)

            # Update trackers AFTER generating features (walk-forward)
            stats_tracker.update(
                home,
                away,
                float(fouls_home),
                float(fouls_away),
                float(home_xg),
                float(away_xg),
            )
            h2h_tracker.update(home, away, float(fouls_total))
            ref_tracker.update(ref_name, home, away, float(fouls_total), mu)

    return rows


# ---------------------------------------------------------------------------
# Build and write Parquet
# ---------------------------------------------------------------------------


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns have correct dtypes for training compatibility.

    Args:
        df: Raw DataFrame from simulation.

    Returns:
        DataFrame with corrected dtypes.
    """
    # String columns
    for col in [
        "home_team",
        "away_team",
        "referee",
        "season",
        "date",
        "intensidad_esperada",
        "riesgo_disciplinario",
    ]:
        df[col] = df[col].astype(str)

    # Bool columns
    df["is_derby"] = df["is_derby"].astype(bool)
    df["has_market_odds"] = df["has_market_odds"].astype(bool)

    # Integer columns
    for col in [
        "matchday",
        "home_rank_curr",
        "away_rank_curr",
        "referee_n_partidos",
        "h2h_partidos",
    ]:
        df[col] = df[col].astype("int64")

    # Float columns: everything else that's numeric
    float_cols = [
        c
        for c in df.columns
        if c
        not in {
            "home_team",
            "away_team",
            "referee",
            "season",
            "date",
            "intensidad_esperada",
            "riesgo_disciplinario",
            "is_derby",
            "has_market_odds",
            "matchday",
            "home_rank_curr",
            "away_rank_curr",
            "referee_n_partidos",
            "h2h_partidos",
        }
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required columns are present with correct dtypes.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or have wrong dtypes.
    """
    required_cols = [
        "home_team",
        "away_team",
        "referee",
        "matchday",
        "season",
        "date",
        "home_fouls_committed_avg",
        "home_fouls_suffered_avg",
        "away_fouls_committed_avg",
        "away_fouls_suffered_avg",
        "home_fouls_committed_curr",
        "away_fouls_committed_curr",
        "home_shots_curr",
        "away_shots_curr",
        "home_corners_curr",
        "away_corners_curr",
        "home_yellows_avg",
        "away_yellows_avg",
        "home_reds_avg",
        "away_reds_avg",
        "fouls_provoked_home",
        "fouls_provoked_away",
        "home_rank_hist",
        "away_rank_hist",
        "home_rank_curr",
        "away_rank_curr",
        "rank_diff_norm",
        "home_xg",
        "away_xg",
        "xg_diff",
        "home_possession",
        "away_possession",
        "xfouls_home",
        "xfouls_away",
        "xfouls_factor_home",
        "xfouls_factor_away",
        "aggressiveness_volume_home",
        "aggressiveness_volume_away",
        "aggressiveness_norm_total",
        "forma_fouls_home",
        "forma_fouls_away",
        "momentum_home",
        "momentum_away",
        "season_phase",
        "is_derby",
        "urgency_home",
        "urgency_away",
        "fatigue_home",
        "fatigue_away",
        "days_rest_home",
        "days_rest_away",
        "referee_mu_permisivo",
        "referee_mu_estricto",
        "referee_sigma_permisivo",
        "referee_sigma_estricto",
        "referee_peso_estricto",
        "referee_n_partidos",
        "ref_home_delta",
        "ref_away_delta",
        "ref_pair_delta_sum",
        "ref_pair_samples",
        "has_market_odds",
        "market_home_win_prob",
        "market_draw_prob",
        "market_away_win_prob",
        "market_favorite_prob",
        "market_balance",
        "market_entropy",
        "market_ou25_over_prob",
        "market_ou25_under_prob",
        "foul_market_prob_over",
        "foul_market_implied_mean",
        "intensidad_esperada",
        "riesgo_disciplinario",
        "pace_index_curr",
        "h2h_faltas_media",
        "h2h_partidos",
        "fouls_total",
        "fouls_home",
        "fouls_away",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type checks
    assert df["is_derby"].dtype == bool, (
        f"is_derby must be bool, got {df['is_derby'].dtype}"
    )
    assert df["matchday"].dtype == "int64", (
        f"matchday must be int64, got {df['matchday'].dtype}"
    )
    assert df["fouls_total"].dtype == "float64", f"fouls_total must be float64"
    assert (
        df["foul_market_prob_over"].nunique() == 1
        and df["foul_market_prob_over"].iloc[0] == 0.5
    )
    assert (
        df["foul_market_implied_mean"].nunique() == 1
        and df["foul_market_implied_mean"].iloc[0] == 24.5
    )

    logger.info(
        "Schema validation passed — %d columns, %d rows", len(df.columns), len(df)
    )


def simulate(
    n_seasons: int = 7,
    seed: int = 42,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Run the full Monte Carlo simulation and write the Parquet.

    Args:
        n_seasons: Number of seasons to simulate (uses last n from _ALL_SEASONS).
        seed: Random seed for reproducibility.
        output_path: Output path for the Parquet file.

    Returns:
        Generated DataFrame.
    """
    rng = np.random.default_rng(seed)

    seasons = _ALL_SEASONS[-n_seasons:]
    logger.info("Simulating %d seasons: %s", n_seasons, seasons)

    all_rows: list[dict] = []
    h2h_tracker = _H2HTracker()
    ref_tracker = _RefInteractionTracker()

    prev_team_profiles: dict[str, dict[str, float]] | None = None
    prev_ref_profiles: dict[str, dict[str, float]] | None = None

    for season_idx, season in enumerate(seasons):
        logger.info("  Season %s...", season)

        team_profiles = _build_team_profiles(season_idx, rng, prev_team_profiles)
        ref_profiles = _build_referee_profiles(rng, prev_ref_profiles, n_referees=20)

        rows = simulate_season(
            season=season,
            season_idx=season_idx,
            team_profiles=team_profiles,
            ref_profiles=ref_profiles,
            h2h_tracker=h2h_tracker,
            ref_tracker=ref_tracker,
            rng=rng,
        )
        all_rows.extend(rows)
        logger.info("    %d rows generated", len(rows))

        prev_team_profiles = team_profiles
        prev_ref_profiles = ref_profiles

    df = pd.DataFrame(all_rows)
    df = _fix_dtypes(df)

    # Ensure correct column order (80 cols)
    expected_order = [
        "home_team",
        "away_team",
        "referee",
        "matchday",
        "season",
        "date",
        "home_fouls_committed_avg",
        "home_fouls_suffered_avg",
        "away_fouls_committed_avg",
        "away_fouls_suffered_avg",
        "home_fouls_committed_curr",
        "away_fouls_committed_curr",
        "home_shots_curr",
        "away_shots_curr",
        "home_corners_curr",
        "away_corners_curr",
        "home_yellows_avg",
        "away_yellows_avg",
        "home_reds_avg",
        "away_reds_avg",
        "fouls_provoked_home",
        "fouls_provoked_away",
        "home_rank_hist",
        "away_rank_hist",
        "home_rank_curr",
        "away_rank_curr",
        "rank_diff_norm",
        "home_xg",
        "away_xg",
        "xg_diff",
        "home_possession",
        "away_possession",
        "xfouls_home",
        "xfouls_away",
        "xfouls_factor_home",
        "xfouls_factor_away",
        "aggressiveness_volume_home",
        "aggressiveness_volume_away",
        "aggressiveness_norm_total",
        "forma_fouls_home",
        "forma_fouls_away",
        "momentum_home",
        "momentum_away",
        "season_phase",
        "is_derby",
        "urgency_home",
        "urgency_away",
        "fatigue_home",
        "fatigue_away",
        "days_rest_home",
        "days_rest_away",
        "referee_mu_permisivo",
        "referee_mu_estricto",
        "referee_sigma_permisivo",
        "referee_sigma_estricto",
        "referee_peso_estricto",
        "referee_n_partidos",
        "ref_home_delta",
        "ref_away_delta",
        "ref_pair_delta_sum",
        "ref_pair_samples",
        "has_market_odds",
        "market_home_win_prob",
        "market_draw_prob",
        "market_away_win_prob",
        "market_favorite_prob",
        "market_balance",
        "market_entropy",
        "market_ou25_over_prob",
        "market_ou25_under_prob",
        "foul_market_prob_over",
        "foul_market_implied_mean",
        "intensidad_esperada",
        "riesgo_disciplinario",
        "pace_index_curr",
        "h2h_faltas_media",
        "h2h_partidos",
        "fouls_total",
        "fouls_home",
        "fouls_away",
    ]
    # Add any unexpected extra columns at the end
    extra_cols = [c for c in df.columns if c not in expected_order]
    col_order = expected_order + extra_cols
    df = df[[c for c in col_order if c in df.columns]]

    _validate_schema(df)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Wrote Parquet: %s (%d rows x %d cols)",
        output_path,
        len(df),
        len(df.columns),
    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo simulation — La Liga training data"
    )
    p.add_argument(
        "--seasons",
        type=int,
        default=7,
        help="Number of seasons to simulate (default: 7 → ~2660 rows)",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output Parquet path",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point for the simulation script."""
    args = _parse_args()
    output_path = Path(args.output)

    logger.info(
        "Starting Monte Carlo simulation (seed=%d, seasons=%d)", args.seed, args.seasons
    )
    df = simulate(n_seasons=args.seasons, seed=args.seed, output_path=output_path)

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Teams per season: {df.groupby('season')['home_team'].nunique().mean():.0f}")
    print(f"Referees total: {df['referee'].nunique()}")
    print(f"Derbies: {df['is_derby'].sum()} ({df['is_derby'].mean() * 100:.1f}%)")
    print(f"\nTarget stats:")
    print(
        f"  fouls_total: mean={df['fouls_total'].mean():.1f}, std={df['fouls_total'].std():.1f}, "
        f"min={df['fouls_total'].min():.0f}, max={df['fouls_total'].max():.0f}"
    )
    print(
        f"  fouls_home:  mean={df['fouls_home'].mean():.1f}, std={df['fouls_home'].std():.1f}"
    )
    print(
        f"  fouls_away:  mean={df['fouls_away'].mean():.1f}, std={df['fouls_away'].std():.1f}"
    )
    print(f"\nKey feature correlations with fouls_total:")
    for col in [
        "xfouls_home",
        "xfouls_away",
        "aggressiveness_volume_home",
        "aggressiveness_norm_total",
        "referee_mu_estricto",
    ]:
        if col in df.columns:
            r = df[["fouls_total", col]].corr().iloc[0, 1]
            print(f"  fouls_total ~ {col}: r={r:.3f}")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
