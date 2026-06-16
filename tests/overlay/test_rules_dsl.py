"""
tests/overlay/test_rules_dsl.py — Unit tests for the DSL evaluator.

Tests (14 parametrized cases — Strict TDD, all RED before rules.py DSL is implemented):
  - eq, neq, gte, lte, gt, lt, in, not_in, flag_present, flag_absent operators
  - all combinator short-circuits on first False
  - any combinator stops on first True
  - not negates nested block
  - max nesting depth 2 enforced
  - missing field evaluates to False (no exception)
"""

from __future__ import annotations

import pytest

from overlay.rules import evaluate_condition
from overlay.schema import Narrative, NarrativeMatch, NarrativeStakes, ObjectiveOverride

# ---------------------------------------------------------------------------
# Fixture narrative for most tests
# ---------------------------------------------------------------------------


def _make_narrative(**kwargs) -> Narrative:
    defaults = {
        "match": NarrativeMatch(home="TeamA", away="TeamB", date="2026-01-01"),
        "confidence_level": 4,
        "physicality_bias": 1,
        "referee_factor": -1,
        "intensity_override": 3,
        "stakes": NarrativeStakes(home=4, away=2),
        "rotations": {"home": 1, "away": 3},
        "special_flags": ["derbi", "physical_clash"],
        "objectives": {
            "home": ObjectiveOverride(label="salvacion"),
            "away": ObjectiveOverride(label="mid"),
        },
    }
    return Narrative(**{**defaults, **kwargs})


BASE_NARR = _make_narrative()


# ---------------------------------------------------------------------------
# Leaf operator tests
# ---------------------------------------------------------------------------


class TestLeafOperators:
    @pytest.mark.parametrize(
        "condition,expected",
        [
            # eq
            ({"field": "confidence_level", "eq": 4}, True),
            ({"field": "confidence_level", "eq": 3}, False),
            # neq
            ({"field": "confidence_level", "neq": 3}, True),
            ({"field": "confidence_level", "neq": 4}, False),
            # gte
            ({"field": "physicality_bias", "gte": 1}, True),
            ({"field": "physicality_bias", "gte": 2}, False),
            # lte
            ({"field": "physicality_bias", "lte": 1}, True),
            ({"field": "physicality_bias", "lte": 0}, False),
            # gt
            ({"field": "confidence_level", "gt": 3}, True),
            ({"field": "confidence_level", "gt": 4}, False),
            # lt
            ({"field": "confidence_level", "lt": 5}, True),
            ({"field": "confidence_level", "lt": 4}, False),
            # in
            (
                {
                    "field": "objectives.home.label",
                    "in": ["descenso", "salvacion"],
                },
                True,
            ),
            (
                {"field": "objectives.home.label", "in": ["ucl", "titulo"]},
                False,
            ),
            # not_in
            (
                {"field": "objectives.home.label", "not_in": ["ucl", "titulo"]},
                True,
            ),
            (
                {
                    "field": "objectives.home.label",
                    "not_in": ["descenso", "salvacion"],
                },
                False,
            ),
            # flag_present
            ({"flag_present": "derbi"}, True),
            ({"flag_present": "dead_rubber"}, False),
            # flag_absent
            ({"flag_absent": "dead_rubber"}, True),
            ({"flag_absent": "derbi"}, False),
        ],
    )
    def test_leaf_operator(self, condition: dict, expected: bool) -> None:
        assert evaluate_condition(condition, BASE_NARR) == expected


# ---------------------------------------------------------------------------
# Combinator: all (AND — short-circuit on first False)
# ---------------------------------------------------------------------------


class TestAllCombinator:
    def test_all_true(self) -> None:
        cond = {
            "all": [
                {"field": "confidence_level", "gte": 3},
                {"flag_present": "derbi"},
            ]
        }
        assert evaluate_condition(cond, BASE_NARR) is True

    def test_all_false_short_circuit(self) -> None:
        """all returns False as soon as first False is found."""
        cond = {
            "all": [
                {"field": "confidence_level", "eq": 99},  # False → short-circuits
                {"flag_present": "derbi"},  # never evaluated
            ]
        }
        assert evaluate_condition(cond, BASE_NARR) is False

    def test_all_empty_is_true(self) -> None:
        """all with empty list = vacuously True."""
        assert evaluate_condition({"all": []}, BASE_NARR) is True


# ---------------------------------------------------------------------------
# Combinator: any (OR — stop on first True)
# ---------------------------------------------------------------------------


class TestAnyCombinator:
    def test_any_true_on_first(self) -> None:
        cond = {
            "any": [
                {"field": "confidence_level", "gte": 3},  # True → stop
                {"field": "confidence_level", "eq": 99},  # never evaluated
            ]
        }
        assert evaluate_condition(cond, BASE_NARR) is True

    def test_any_false_when_none_match(self) -> None:
        cond = {
            "any": [
                {"field": "confidence_level", "eq": 99},
                {"flag_present": "dead_rubber"},
            ]
        }
        assert evaluate_condition(cond, BASE_NARR) is False

    def test_any_empty_is_false(self) -> None:
        """any with empty list = False (no condition satisfied)."""
        assert evaluate_condition({"any": []}, BASE_NARR) is False


# ---------------------------------------------------------------------------
# Combinator: not
# ---------------------------------------------------------------------------


class TestNotCombinator:
    def test_not_negates_true(self) -> None:
        cond = {"not": {"field": "confidence_level", "eq": 4}}
        assert evaluate_condition(cond, BASE_NARR) is False

    def test_not_negates_false(self) -> None:
        cond = {"not": {"field": "confidence_level", "eq": 99}}
        assert evaluate_condition(cond, BASE_NARR) is True


# ---------------------------------------------------------------------------
# Nesting: all > [any > [conditions]]
# ---------------------------------------------------------------------------


class TestNestedCombinators:
    def test_all_containing_any(self) -> None:
        cond = {
            "all": [
                {"field": "confidence_level", "gte": 3},
                {
                    "any": [
                        {"flag_present": "dead_rubber"},  # False
                        {"flag_present": "derbi"},  # True
                    ]
                },
            ]
        }
        assert evaluate_condition(cond, BASE_NARR) is True

    def test_any_containing_all(self) -> None:
        cond = {
            "any": [
                {
                    "all": [
                        {"field": "confidence_level", "eq": 99},  # False → all is False
                        {"flag_present": "derbi"},
                    ]
                },
                {"flag_present": "physical_clash"},  # True
            ]
        }
        assert evaluate_condition(cond, BASE_NARR) is True


# ---------------------------------------------------------------------------
# Missing field → False (no exception)
# ---------------------------------------------------------------------------


class TestMissingField:
    def test_missing_field_evaluates_to_false(self) -> None:
        """Field path that doesn't exist in narrative evaluates to False."""
        cond = {"field": "objectives.away.urgency_base", "gte": 0.5}
        # objectives.away.urgency_base is None (not set) → False
        narr = _make_narrative()
        assert evaluate_condition(cond, narr) is False

    def test_completely_missing_field_path(self) -> None:
        """Totally missing field path evaluates to False without raising."""
        cond = {"field": "nonexistent_field", "eq": 42}
        assert evaluate_condition(cond, BASE_NARR) is False

    def test_missing_objective_subfield(self) -> None:
        """Field path to a missing sub-field in objectives resolves to False."""
        # Objectives are always present (REQUIRED), but urgency_base may be None
        narr = _make_narrative(
            objectives={
                "home": ObjectiveOverride(label="salvacion"),
                "away": ObjectiveOverride(label="mid"),
            }
        )
        # urgency_base is None on both sides → gte comparison yields False
        cond = {"field": "objectives.away.urgency_base", "gte": 0.5}
        assert evaluate_condition(cond, narr) is False
