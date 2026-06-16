"""
staking.loader — Load and validate overlay/staking.yaml.

Validates the staking curve configuration per REQ-9.3:
  - No overlapping thresholds
  - Stakes in range 1-10
  - bankroll_share_per_stake_unit * max_stake <= 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["StakingConfigError", "StakingCurve", "load_staking_curve"]


class StakingConfigError(ValueError):
    """Raised when staking.yaml fails validation."""


@dataclass
class StakingCurve:
    """Validated staking curve configuration.

    Args:
        bankroll_share_per_stake_unit: Fraction of bankroll per stake unit (e.g. 0.06).
        no_bet_below_edge: Minimum edge to place any bet (e.g. 0.05).
        edge_thresholds: List of threshold dicts with edge_min, edge_max, stake.
    """

    bankroll_share_per_stake_unit: float
    no_bet_below_edge: float
    edge_thresholds: list[dict[str, Any]] = field(default_factory=list)


def load_staking_curve(path: Path) -> StakingCurve:
    """Load and validate overlay/staking.yaml.

    Args:
        path: Path to the staking YAML file.

    Returns:
        Validated StakingCurve dataclass.

    Raises:
        FileNotFoundError: If the file does not exist.
        StakingConfigError: If the configuration fails validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Staking config not found: {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise StakingConfigError(f"YAML parse error in {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise StakingConfigError(
            f"Staking config must be a YAML mapping, got {type(data).__name__}"
        )

    share = float(data.get("bankroll_share_per_stake_unit", 0))
    no_bet = float(data.get("no_bet_below_edge", 0))
    thresholds: list[dict[str, Any]] = list(data.get("edge_thresholds") or [])

    _validate(share, no_bet, thresholds)

    return StakingCurve(
        bankroll_share_per_stake_unit=share,
        no_bet_below_edge=no_bet,
        edge_thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(
    share: float,
    no_bet: float,
    thresholds: list[dict[str, Any]],
) -> None:
    """Validate staking curve parameters.

    Raises:
        StakingConfigError: On any validation failure.
    """
    if not thresholds:
        raise StakingConfigError("edge_thresholds must not be empty")

    # Validate individual stakes
    max_stake = 0
    for i, t in enumerate(thresholds):
        stake = int(t.get("stake", 0))
        if not (1 <= stake <= 10):
            raise StakingConfigError(
                f"Threshold {i}: stake must be in [1, 10], got {stake}"
            )
        max_stake = max(max_stake, stake)

    # Validate bankroll share * max stake <= 1.0
    total = share * max_stake
    if total > 1.0:
        raise StakingConfigError(
            f"{total * 100:.0f}% bankroll allocation is invalid "
            f"(share={share} * max_stake={max_stake} = {total:.2f} > 1.0)"
        )

    # Validate no overlapping thresholds
    for i in range(len(thresholds) - 1):
        curr = thresholds[i]
        nxt = thresholds[i + 1]
        curr_max = float(curr.get("edge_max", 1.0))
        nxt_min = float(nxt.get("edge_min", 0.0))
        if nxt_min < curr_max:
            raise StakingConfigError(
                f"Threshold overlap between index {i} (edge_max={curr_max}) "
                f"and index {i + 1} (edge_min={nxt_min})"
            )
