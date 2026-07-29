"""Domain entity invariant tests — REQ-1."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import pytest

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.match import Match
from HWFP.core.domain.match_prediction import MatchPrediction
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import HoldoutMetrics, ModelManifest
from HWFP.core.domain.model_status import ModelStatus
from HWFP.core.domain.narrative import Narrative
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.stake_result import StakeResult

_DT = datetime(2026, 6, 16, 20, 0, 0)
_MID = ModelId(value="foul-model-001")
_PMF = FoulPMF(pmf=(0.5, 0.5), bin_edges=(0, 22, 40))
_METRICS = HoldoutMetrics(nll=0.5, brier=0.18, calibration_ece=0.03)


def _all_frozen_entities() -> list[object]:
    return [
        Match(
            match_id="M1",
            home_team_id="T_HOME",
            away_team_id="T_AWAY",
            kickoff=_DT,
            referee_id="R1",
            competition_id="laliga",
        ),
        _PMF,
        Odds(
            match_id="M1",
            market="fouls_over_under",
            line=22.5,
            side="over",
            decimal=1.95,
            bookmaker="codere",
            fetched_at=_DT,
        ),
        MatchPrediction(match_id="M1", pmf=_PMF, model_id=_MID, generated_at=_DT),
        EVResult(
            match_id="M1",
            market="fouls_over_under",
            line=22.5,
            side="over",
            fair_prob=0.6,
            book_prob=0.51,
            ev=0.17,
        ),
        StakeResult(
            match_id="M1",
            market="fouls_over_under",
            stake=10.0,
            kelly_fraction=0.25,
            bankroll_used=40.0,
        ),
        Narrative(match_id="M1", text="High fouls expected.", confidence=0.8),
        _MID,
        ModelManifest(
            model_id=_MID,
            trained_at=_DT,
            git_sha="abc123",
            dataset_hash="hash-001",
            dataset_rows=100,
            metrics_holdout=_METRICS,
            gates_passed=("nll_ok",),
            status=ModelStatus.CANDIDATE,
        ),
        _METRICS,
    ]


def test_entities_are_frozen() -> None:
    for entity in _all_frozen_entities():
        first_field = next(iter(entity.__dataclass_fields__))  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            setattr(entity, first_field, "mutated")


def test_entities_have_no_io_imports() -> None:
    domain_path = Path(__file__).resolve().parents[2] / "core" / "domain"
    forbidden = {"os", "io", "pathlib", "requests", "supabase", "torch"}
    violations: list[tuple[str, str]] = []
    for py_file in domain_path.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in forbidden:
                        violations.append((str(py_file), alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top in forbidden:
                        violations.append((str(py_file), node.module))
    assert not violations, f"I/O imports found in core/domain/: {violations}"
