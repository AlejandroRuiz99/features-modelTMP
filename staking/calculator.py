"""
staking.calculator — Edge → stake → euros computation.

D6 design decision: kelly_scale is applied BEFORE stake quantize.
Specifically: stake_n is determined by edge alone (from thresholds).
kelly_scale then multiplies bank_share_raw → bank_share_final → euros.
The stake level (stake_n) is NEVER modified by kelly_scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from staking.loader import StakingCurve

__all__ = ["StakeResult", "compute_stake"]


@dataclass
class StakeResult:
    """Result of staking calculation for a single pick.

    Fields:
        edge: Input edge (decimal, e.g. 0.12 for 12%).
        stake_n: Stake level 0..10 from threshold lookup (0 = no bet).
        bank_share_raw: stake_n * bankroll_share_per_stake_unit (before kelly).
        kelly_scale: Kelly scaling factor applied from overlay rules.
        bank_share_final: bank_share_raw * kelly_scale (after kelly).
        euros: bank_share_final * bankroll.
    """

    edge: float
    stake_n: int
    bank_share_raw: float
    kelly_scale: float
    bank_share_final: float
    euros: float


def compute_stake(
    edge: float,
    kelly_scale: float,
    bankroll: float,
    curve: StakingCurve,
) -> StakeResult:
    """Compute stake from edge, applying kelly_scale before euro quantization.

    D6 timing:
      1. edge → lookup thresholds → stake_n (0 if edge < no_bet_below_edge)
      2. bank_share_raw = stake_n * bankroll_share_per_stake_unit
      3. bank_share_final = bank_share_raw * kelly_scale
      4. euros = bank_share_final * bankroll

    Args:
        edge: Prediction edge (decimal fraction, e.g. 0.12 for 12%).
        kelly_scale: Kelly scaling factor from overlay rules (e.g. 0.85).
        bankroll: Current bankroll in euros.
        curve: Validated StakingCurve configuration.

    Returns:
        StakeResult with all fields populated.
    """
    stake_n = _lookup_stake(edge, curve)

    bank_share_raw = stake_n * curve.bankroll_share_per_stake_unit
    bank_share_final = bank_share_raw * kelly_scale
    euros = bank_share_final * bankroll

    return StakeResult(
        edge=edge,
        stake_n=stake_n,
        bank_share_raw=bank_share_raw,
        kelly_scale=kelly_scale,
        bank_share_final=bank_share_final,
        euros=euros,
    )


def _lookup_stake(edge: float, curve: StakingCurve) -> int:
    """Look up stake_n from the edge threshold table.

    Returns 0 if edge is below no_bet_below_edge or matches no threshold.
    """
    if edge < curve.no_bet_below_edge:
        return 0

    # Walk thresholds — first match wins
    for threshold in curve.edge_thresholds:
        edge_min = float(threshold.get("edge_min", 0))
        edge_max = float(threshold.get("edge_max", 1.0))
        stake = int(threshold.get("stake", 0))

        if edge >= edge_min and edge < edge_max:
            return stake

    # Edge is above all thresholds: return the maximum stake
    if curve.edge_thresholds:
        return int(curve.edge_thresholds[-1].get("stake", 0))

    return 0
