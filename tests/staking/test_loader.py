"""
tests/staking/test_loader.py — TDD for staking.loader

Tests (T3.1):
  1. Valid overlay/staking.yaml loads without error.
  2. Overlapping thresholds raises StakingConfigError.
  3. Stake outside 1-10 raises StakingConfigError.
  4. bankroll_share_per_stake_unit * max_stake > 1.0 raises StakingConfigError.
  5. Loaded StakingCurve has expected fields.
  6. no_bet_below_edge is preserved.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from staking.loader import StakingConfigError, StakingCurve, load_staking_curve

# ---------------------------------------------------------------------------
# YAML fixtures
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
bankroll_share_per_stake_unit: 0.06
no_bet_below_edge: 0.05
edge_thresholds:
  - {edge_min: 0.05, edge_max: 0.07, stake: 2}
  - {edge_min: 0.07, edge_max: 0.10, stake: 4}
  - {edge_min: 0.10, edge_max: 0.13, stake: 6}
  - {edge_min: 0.13, edge_max: 0.16, stake: 8}
  - {edge_min: 0.16, edge_max: 1.00, stake: 10}
""")

_OVERLAPPING_YAML = textwrap.dedent("""\
bankroll_share_per_stake_unit: 0.06
no_bet_below_edge: 0.05
edge_thresholds:
  - {edge_min: 0.05, edge_max: 0.10, stake: 2}
  - {edge_min: 0.08, edge_max: 0.15, stake: 4}
""")

_STAKE_TOO_HIGH_YAML = textwrap.dedent("""\
bankroll_share_per_stake_unit: 0.06
no_bet_below_edge: 0.05
edge_thresholds:
  - {edge_min: 0.05, edge_max: 0.10, stake: 2}
  - {edge_min: 0.10, edge_max: 1.00, stake: 15}
""")

_SHARE_TOO_HIGH_YAML = textwrap.dedent("""\
bankroll_share_per_stake_unit: 0.15
no_bet_below_edge: 0.05
edge_thresholds:
  - {edge_min: 0.05, edge_max: 0.10, stake: 2}
  - {edge_min: 0.10, edge_max: 1.00, stake: 10}
""")

_STAKE_ZERO_YAML = textwrap.dedent("""\
bankroll_share_per_stake_unit: 0.06
no_bet_below_edge: 0.05
edge_thresholds:
  - {edge_min: 0.05, edge_max: 1.00, stake: 0}
""")


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestValidLoad:
    def test_valid_yaml_loads_without_error(self, tmp_path: Path) -> None:
        """Valid staking YAML loads successfully."""
        p = _write_yaml(tmp_path / "staking.yaml", _VALID_YAML)
        curve = load_staking_curve(p)
        assert isinstance(curve, StakingCurve)

    def test_bankroll_share_preserved(self, tmp_path: Path) -> None:
        """bankroll_share_per_stake_unit is correctly loaded."""
        p = _write_yaml(tmp_path / "staking.yaml", _VALID_YAML)
        curve = load_staking_curve(p)
        assert curve.bankroll_share_per_stake_unit == pytest.approx(0.06)

    def test_no_bet_below_edge_preserved(self, tmp_path: Path) -> None:
        """no_bet_below_edge is correctly loaded."""
        p = _write_yaml(tmp_path / "staking.yaml", _VALID_YAML)
        curve = load_staking_curve(p)
        assert curve.no_bet_below_edge == pytest.approx(0.05)

    def test_five_thresholds_loaded(self, tmp_path: Path) -> None:
        """All 5 thresholds are loaded."""
        p = _write_yaml(tmp_path / "staking.yaml", _VALID_YAML)
        curve = load_staking_curve(p)
        assert len(curve.edge_thresholds) == 5

    def test_threshold_fields_correct(self, tmp_path: Path) -> None:
        """First threshold has correct edge_min, edge_max, stake."""
        p = _write_yaml(tmp_path / "staking.yaml", _VALID_YAML)
        curve = load_staking_curve(p)
        first = curve.edge_thresholds[0]
        assert first["edge_min"] == pytest.approx(0.05)
        assert first["edge_max"] == pytest.approx(0.07)
        assert first["stake"] == 2


class TestValidationErrors:
    def test_overlapping_thresholds_raises(self, tmp_path: Path) -> None:
        """Overlapping edge thresholds raise StakingConfigError."""
        p = _write_yaml(tmp_path / "staking.yaml", _OVERLAPPING_YAML)
        with pytest.raises(StakingConfigError, match="overlap"):
            load_staking_curve(p)

    def test_stake_above_10_raises(self, tmp_path: Path) -> None:
        """Stake > 10 raises StakingConfigError."""
        p = _write_yaml(tmp_path / "staking.yaml", _STAKE_TOO_HIGH_YAML)
        with pytest.raises(StakingConfigError):
            load_staking_curve(p)

    def test_stake_zero_raises(self, tmp_path: Path) -> None:
        """Stake of 0 raises StakingConfigError."""
        p = _write_yaml(tmp_path / "staking.yaml", _STAKE_ZERO_YAML)
        with pytest.raises(StakingConfigError):
            load_staking_curve(p)

    def test_share_times_max_stake_gt_1_raises(self, tmp_path: Path) -> None:
        """bankroll_share * max_stake > 1.0 raises StakingConfigError."""
        p = _write_yaml(tmp_path / "staking.yaml", _SHARE_TOO_HIGH_YAML)
        with pytest.raises(StakingConfigError):
            load_staking_curve(p)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_staking_curve(tmp_path / "nonexistent.yaml")
