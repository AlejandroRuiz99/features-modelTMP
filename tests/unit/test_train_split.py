"""Unit tests for the 3-way temporal split helper in scripts/train.py."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from scripts.train import temporal_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_season_df() -> pd.DataFrame:
    """Synthetic DataFrame with 3 distinct seasons and a date column.

    Seasons:
    - "2023-24": 100 rows (dates 2024-01-01 … 2024-04-10)
    - "2024-25": 120 rows (dates 2025-01-01 … 2025-04-30)
    - "2025-26": 80 rows  (dates 2026-01-01 … 2026-03-21)
    Total: 300 rows — strictly ordered by date within each season.
    """
    seasons = ["2023-24"] * 100 + ["2024-25"] * 120 + ["2025-26"] * 80
    dates = (
        list(pd.date_range("2024-01-01", periods=100, freq="D"))
        + list(pd.date_range("2025-01-01", periods=120, freq="D"))
        + list(pd.date_range("2026-01-01", periods=80, freq="D"))
    )
    return pd.DataFrame({"season": seasons, "date": dates, "fouls_total": 25})


@pytest.fixture
def two_season_df() -> pd.DataFrame:
    """Synthetic DataFrame with only 2 seasons — missing '2025-26'."""
    seasons = ["2023-24"] * 100 + ["2024-25"] * 120
    dates = list(pd.date_range("2024-01-01", periods=100, freq="D")) + list(
        pd.date_range("2025-01-01", periods=120, freq="D")
    )
    return pd.DataFrame({"season": seasons, "date": dates, "fouls_total": 25})


# ---------------------------------------------------------------------------
# Tests: happy path — 3-way split
# ---------------------------------------------------------------------------


class TestTemporalSplitHappyPath:
    """Tests for temporal_split with all 3 seasons present."""

    def test_returns_three_dataframes(self, three_season_df: pd.DataFrame) -> None:
        """temporal_split returns exactly 3 DataFrames."""
        result = temporal_split(
            three_season_df,
            train_seasons=["2023-24", "2024-25"],
            tune_season="2025-26",
            test_season="2025-26",
        )
        assert len(result) == 3
        train, tune, test = result
        assert isinstance(train, pd.DataFrame)
        assert isinstance(tune, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_row_counts_sum_to_total(self, three_season_df: pd.DataFrame) -> None:
        """len(train) + len(tune) + len(test) == len(df)."""
        train, tune, test = temporal_split(
            three_season_df,
            train_seasons=["2023-24", "2024-25"],
            tune_season="2025-26",
            test_season="2025-26",
        )
        assert len(train) + len(tune) + len(test) == len(three_season_df)

    def test_sets_are_non_overlapping(self, three_season_df: pd.DataFrame) -> None:
        """No shared indices between train, tune, and test."""
        train, tune, test = temporal_split(
            three_season_df,
            train_seasons=["2023-24", "2024-25"],
            tune_season="2025-26",
            test_season="2025-26",
        )
        train_idx = set(train.index)
        tune_idx = set(tune.index)
        test_idx = set(test.index)

        assert train_idx.isdisjoint(tune_idx), "train and tune share indices"
        assert train_idx.isdisjoint(test_idx), "train and test share indices"
        assert tune_idx.isdisjoint(test_idx), "tune and test share indices"

    def test_temporal_ordering_preserved(self, three_season_df: pd.DataFrame) -> None:
        """All train dates < all tune dates < all test dates."""
        train, tune, test = temporal_split(
            three_season_df,
            train_seasons=["2023-24", "2024-25"],
            tune_season="2025-26",
            test_season="2025-26",
        )
        max_train_date = train["date"].max()
        min_tune_date = tune["date"].min()
        min_test_date = test["date"].min()

        assert max_train_date < min_tune_date, (
            f"Train has date {max_train_date} which is not before "
            f"earliest tune date {min_tune_date}"
        )
        assert min_tune_date <= min_test_date, (
            f"Tune date {min_tune_date} is not before test date {min_test_date}"
        )

    def test_train_contains_correct_seasons(
        self, three_season_df: pd.DataFrame
    ) -> None:
        """Train split contains exactly the requested train_seasons."""
        train, _, _ = temporal_split(
            three_season_df,
            train_seasons=["2023-24", "2024-25"],
            tune_season="2025-26",
            test_season="2025-26",
        )
        assert set(train["season"].unique()) == {"2023-24", "2024-25"}

    def test_train_row_count(self, three_season_df: pd.DataFrame) -> None:
        """Train split has 100 + 120 = 220 rows for this fixture."""
        train, _, _ = temporal_split(
            three_season_df,
            train_seasons=["2023-24", "2024-25"],
            tune_season="2025-26",
            test_season="2025-26",
        )
        assert len(train) == 220


# ---------------------------------------------------------------------------
# Tests: edge case — missing season → empty DataFrame + warning
# ---------------------------------------------------------------------------


class TestTemporalSplitMissingSeason:
    """Tests for temporal_split when a requested season is absent."""

    def test_missing_tune_season_returns_empty_tune(
        self, two_season_df: pd.DataFrame
    ) -> None:
        """If tune_season is absent, tune is an empty DataFrame."""
        train, tune, test = temporal_split(
            two_season_df,
            train_seasons=["2023-24"],
            tune_season="2025-26",  # absent
            test_season="2024-25",
        )
        assert isinstance(tune, pd.DataFrame)
        assert len(tune) == 0

    def test_missing_season_emits_warning(self, two_season_df: pd.DataFrame) -> None:
        """If a requested season is absent, a UserWarning is emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            temporal_split(
                two_season_df,
                train_seasons=["2023-24"],
                tune_season="2025-26",  # absent
                test_season="2024-25",
            )
        warning_messages = [str(warning.message) for warning in w]
        assert any("2025-26" in msg for msg in warning_messages), (
            f"Expected a warning mentioning '2025-26'. Got: {warning_messages}"
        )

    def test_missing_test_season_returns_empty_test(
        self, two_season_df: pd.DataFrame
    ) -> None:
        """If test_season is absent, test is an empty DataFrame."""
        _, _, test = temporal_split(
            two_season_df,
            train_seasons=["2023-24"],
            tune_season="2024-25",
            test_season="2025-26",  # absent
        )
        assert isinstance(test, pd.DataFrame)
        assert len(test) == 0
