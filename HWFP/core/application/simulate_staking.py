"""SimulateStakingUseCase — rolling bankroll simulation over an EV sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.stake_result import StakeResult
from HWFP.core.ports.staking_calculator import StakingCalculator


@dataclass
class SimulateInput:
    ev_sequence: Tuple[EVResult, ...]
    initial_bankroll: float


@dataclass
class SimulateOutput:
    final_bankroll: float
    stakes: Tuple[StakeResult, ...]


class SimulateStakingUseCase:
    def __init__(self, staking: StakingCalculator) -> None:
        self._staking = staking

    def execute(self, inp: SimulateInput) -> SimulateOutput:
        bankroll = inp.initial_bankroll
        stakes = []
        for ev in inp.ev_sequence:
            stake_result = self._staking.compute(ev, bankroll)
            stakes.append(stake_result)
            bankroll = bankroll + stake_result.stake * ev.ev
        return SimulateOutput(
            final_bankroll=bankroll,
            stakes=tuple(stakes),
        )
