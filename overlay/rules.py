"""
overlay.rules — Rule catalog loading, DSL condition evaluation, and aggregation.

DSL Option B: declarative YAML conditions with predefined operators.
No eval(), no exec(), no Python expression evaluation.
Operators: eq, neq, gte, lte, gt, lt, in, not_in, flag_present, flag_absent
Combinators: all, any, not
Field path syntax: dotted path resolved against Narrative dataclass.
Missing fields → False (no exception).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from overlay.schema import Narrative

__all__ = [
    "AggregatedEffect",
    "Rule",
    "RuleEffect",
    "aggregate_effects",
    "evaluate_condition",
    "evaluate_rule",
    "load_catalog",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OPERATORS = frozenset(
    {
        "eq",
        "neq",
        "gte",
        "lte",
        "gt",
        "lt",
        "in",
        "not_in",
        "flag_present",
        "flag_absent",
    }
)

VALID_COMBINATORS = frozenset({"all", "any", "not"})

VALID_EFFECT_TYPES = frozenset({"delta_fouls", "variance_scale", "kelly_scale"})

VALID_DIRECTIONS = frozenset({"up", "down", "volatility"})

# Magnitude bounds (REQ-5.4)
MAX_DELTA_FOULS = 1.0  # |delta_fouls| <= 1.0 per rule
VARIANCE_SCALE_MIN = 1.0  # variance_scale >= 1.0 per rule
VARIANCE_SCALE_MAX = 1.5  # variance_scale <= 1.5 per rule
KELLY_SCALE_MIN = 0.25  # kelly_scale >= 0.25 per rule
KELLY_SCALE_MAX = 1.0  # kelly_scale <= 1.0 per rule

# Aggregate cap for delta_fouls (enforced at aggregation time)
AGGREGATE_DELTA_CAP = 2.0

# Aggregate variance_scale bounds [1/1.5, 1.5]
AGGREGATE_VARIANCE_MIN = 1.0 / 1.5
AGGREGATE_VARIANCE_MAX = 1.5

# Aggregate kelly_scale bounds
AGGREGATE_KELLY_MIN = 0.25
AGGREGATE_KELLY_MAX = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RuleEffect:
    """Effect of a fired rule on the prediction."""

    delta_fouls: float = 0.0
    variance_scale: float = 1.0
    kelly_scale: float = 1.0


@dataclass
class Rule:
    """A single rule from the catalog."""

    id: str
    description: str
    enabled: bool
    direction: str
    when: dict[str, Any]
    effect: RuleEffect


@dataclass
class AggregatedEffect:
    """Aggregated effect of all fired rules."""

    delta_fouls: float  # After directional gate + cap
    variance_scale: float  # Product of all variance_scales, clipped
    kelly_scale: float  # Min of all kelly_scales, clipped
    raw_delta_sum: float  # Before gate/cap (for logging)
    directional_gate_passed: bool
    same_direction_count: int  # How many rules had same sign as aggregate


# ---------------------------------------------------------------------------
# Field path resolver
# ---------------------------------------------------------------------------


def _resolve_field(narrative: Narrative, path: str) -> Any:
    """Resolve a dotted field path against a Narrative.

        Returns None if any segment is missing — callers treat None as False.

        Supported paths (per design):
    - Top-level: confidence_level, physicality_bias, referee_factor,
                        intensity_override, special_flags, notes
          - objectives.home.label, objectives.home.urgency_base
          - objectives.away.label, objectives.away.urgency_base
          - stakes.home, stakes.away
          - rotations.home, rotations.away
          - match.home, match.away, match.date
    """
    parts = path.split(".")

    # Level 1: top-level attribute
    obj: Any = narrative
    for part in parts:
        if obj is None:
            return None

        # dict access (for rotations which is dict[str, int])
        if isinstance(obj, dict):
            obj = obj.get(part)
            continue

        # dataclass / object attribute access
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    return obj


# ---------------------------------------------------------------------------
# DSL condition evaluator
# ---------------------------------------------------------------------------


def evaluate_condition(condition: dict[str, Any], narrative: Narrative) -> bool:
    """Evaluate a single DSL condition block against a narrative.

    Args:
        condition: A dict representing a DSL condition or combinator.
        narrative: The Narrative to evaluate against.

    Returns:
        True if the condition is satisfied, False otherwise.
        Missing fields always evaluate to False (no exception raised).

    Raises:
        ValueError: If the condition has an unknown operator or structure
            (detected at load time; runtime evaluation should never raise).
    """
    # --- Combinators ---
    if "all" in condition:
        return _eval_all(condition["all"], narrative)
    if "any" in condition:
        return _eval_any(condition["any"], narrative)
    if "not" in condition:
        inner = condition["not"]
        return not evaluate_condition(inner, narrative)

    # --- Flag shortcuts ---
    if "flag_present" in condition:
        flag = condition["flag_present"]
        return flag in (narrative.special_flags or [])
    if "flag_absent" in condition:
        flag = condition["flag_absent"]
        return flag not in (narrative.special_flags or [])

    # --- Field-based operators ---
    if "field" in condition:
        return _eval_field_condition(condition, narrative)

    # Unknown structure — treat as False at runtime
    return False


def _eval_all(conditions: list[dict[str, Any]], narrative: Narrative) -> bool:
    """AND combinator — short-circuits on first False (vacuously True for [])."""
    return all(evaluate_condition(cond, narrative) for cond in conditions)


def _eval_any(conditions: list[dict[str, Any]], narrative: Narrative) -> bool:
    """OR combinator — short-circuits on first True (vacuously False for [])."""
    return any(evaluate_condition(cond, narrative) for cond in conditions)


def _eval_field_condition(condition: dict[str, Any], narrative: Narrative) -> bool:
    """Evaluate a field-based operator condition."""
    path = condition["field"]
    value = _resolve_field(narrative, path)

    # Determine which operator is present
    for op in VALID_OPERATORS:
        if op == "flag_present" or op == "flag_absent":
            continue  # handled separately
        if op in condition:
            operand = condition[op]
            return _apply_operator(op, value, operand)

    # No valid operator found
    return False


def _apply_operator(op: str, value: Any, operand: Any) -> bool:
    """Apply a comparison operator. Returns False if value is None."""
    if value is None:
        return False

    try:
        if op == "eq":
            return value == operand
        if op == "neq":
            return value != operand
        if op == "gte":
            return float(value) >= float(operand)
        if op == "lte":
            return float(value) <= float(operand)
        if op == "gt":
            return float(value) > float(operand)
        if op == "lt":
            return float(value) < float(operand)
        if op == "in":
            return value in operand
        if op == "not_in":
            return value not in operand
    except (TypeError, ValueError):
        return False

    return False


# ---------------------------------------------------------------------------
# Rule evaluator
# ---------------------------------------------------------------------------


def evaluate_rule(rule: Rule, narrative: Narrative) -> bool:
    """Evaluate a rule's trigger condition against a narrative.

    Args:
        rule: The Rule to evaluate.
        narrative: The Narrative to evaluate against.

    Returns:
        True if the rule fires (trigger satisfied and rule is enabled).
    """
    if not rule.enabled:
        return False
    return evaluate_condition(rule.when, narrative)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_effects(
    fired_rules: list[Rule],
    confidence_level: int,
) -> AggregatedEffect:
    """Aggregate effects from all fired rules.

    Args:
        fired_rules: Rules whose trigger evaluated to True.
        confidence_level: Narrative confidence level (1..5).

    Returns:
        AggregatedEffect with delta_fouls (after gate + cap),
        variance_scale (product, clipped), kelly_scale (min, clipped).

    Directional gate (REQ-3.4):
        delta_fouls = 0 if confidence_level < 3
        delta_fouls = 0 if fewer than 2 rules share the aggregate sign
        variance_scale and kelly_scale ALWAYS apply (non-directional).
    """
    if not fired_rules:
        return AggregatedEffect(
            delta_fouls=0.0,
            variance_scale=1.0,
            kelly_scale=1.0,
            raw_delta_sum=0.0,
            directional_gate_passed=False,
            same_direction_count=0,
        )

    # --- delta_fouls: sum then clip ---
    raw_delta_sum = sum(r.effect.delta_fouls for r in fired_rules)

    # Directional gate
    gate_passed = _directional_gate_passes(fired_rules, raw_delta_sum, confidence_level)
    if gate_passed:
        applied_delta = max(
            -AGGREGATE_DELTA_CAP, min(AGGREGATE_DELTA_CAP, raw_delta_sum)
        )
    else:
        applied_delta = 0.0

    # Count rules same direction as aggregate
    agg_sign = _sign(raw_delta_sum)
    same_direction_count = sum(
        1
        for r in fired_rules
        if _sign(r.effect.delta_fouls) == agg_sign and r.effect.delta_fouls != 0.0
    )

    # --- variance_scale: product then clip ---
    product = 1.0
    for r in fired_rules:
        product *= r.effect.variance_scale
    agg_variance = max(AGGREGATE_VARIANCE_MIN, min(AGGREGATE_VARIANCE_MAX, product))

    # --- kelly_scale: minimum then clip ---
    min_kelly = min(r.effect.kelly_scale for r in fired_rules)
    agg_kelly = max(AGGREGATE_KELLY_MIN, min(AGGREGATE_KELLY_MAX, min_kelly))

    return AggregatedEffect(
        delta_fouls=applied_delta,
        variance_scale=agg_variance,
        kelly_scale=agg_kelly,
        raw_delta_sum=raw_delta_sum,
        directional_gate_passed=gate_passed,
        same_direction_count=same_direction_count,
    )


def _directional_gate_passes(
    fired_rules: list[Rule],
    raw_delta_sum: float,
    confidence_level: int,
) -> bool:
    """Return True if the directional shift is allowed to apply.

    Gate conditions (both must hold):
    1. confidence_level >= 3
    2. At least 2 rules have delta_fouls with the same sign as raw_delta_sum
    """
    if confidence_level < 3:
        return False

    if raw_delta_sum == 0.0:
        return False

    agg_sign = _sign(raw_delta_sum)
    same_sign_count = sum(
        1
        for r in fired_rules
        if r.effect.delta_fouls != 0.0 and _sign(r.effect.delta_fouls) == agg_sign
    )
    return same_sign_count >= 2


def _sign(value: float) -> int:
    """Return -1, 0, or +1 for the sign of value."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------


def load_catalog(path: Path | str) -> list[Rule]:
    """Load and validate the rule catalog YAML.

    Args:
        path: Path to rules.yaml.

    Returns:
        List of validated Rule objects (both enabled and disabled).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On YAML parse error, unknown effect type, magnitude out of
            bounds, or unknown DSL operator.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rule catalog not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path.name}: {exc}") from exc

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(f"{path.name}: must contain a top-level 'rules' key")

    raw_rules = data["rules"]
    if not isinstance(raw_rules, list):
        raise ValueError(f"{path.name}: 'rules' must be a list")

    rules: list[Rule] = []
    for idx, raw in enumerate(raw_rules):
        rule = _parse_rule(raw, idx, path.name)
        rules.append(rule)

    return rules


def _parse_rule(raw: Any, idx: int, source: str) -> Rule:
    """Parse and validate a single rule dict."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source}[{idx}]: rule must be a mapping")

    # Required fields
    for key in ("id", "description", "enabled", "direction", "when", "effect"):
        if key not in raw:
            raise ValueError(f"{source}[{idx}]: missing required key '{key}'")

    rule_id = str(raw["id"])
    direction = str(raw["direction"])
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"{source} rule '{rule_id}': invalid direction '{direction}'. "
            f"Valid: {sorted(VALID_DIRECTIONS)}"
        )

    effect = _parse_effect(raw["effect"], rule_id, source)

    # Validate the 'when' condition structure
    _validate_condition_structure(raw["when"], rule_id, source, depth=0)

    return Rule(
        id=rule_id,
        description=str(raw["description"]),
        enabled=bool(raw["enabled"]),
        direction=direction,
        when=raw["when"],
        effect=effect,
    )


def _parse_effect(raw: Any, rule_id: str, source: str) -> RuleEffect:
    """Parse and validate a rule's effect dict."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source} rule '{rule_id}': 'effect' must be a mapping")

    delta_fouls = float(raw.get("delta_fouls", 0.0))
    variance_scale = float(raw.get("variance_scale", 1.0))
    kelly_scale = float(raw.get("kelly_scale", 1.0))

    # Magnitude bounds (REQ-5.4)
    if abs(delta_fouls) > MAX_DELTA_FOULS:
        raise ValueError(
            f"{source} rule '{rule_id}': |delta_fouls| = {abs(delta_fouls):.3f} "
            f"exceeds max {MAX_DELTA_FOULS}. Use values in [-1.0, +1.0]."
        )
    if not (VARIANCE_SCALE_MIN <= variance_scale <= VARIANCE_SCALE_MAX):
        raise ValueError(
            f"{source} rule '{rule_id}': variance_scale = {variance_scale:.3f} "
            f"out of bounds [{VARIANCE_SCALE_MIN}, {VARIANCE_SCALE_MAX}]."
        )
    if not (KELLY_SCALE_MIN <= kelly_scale <= KELLY_SCALE_MAX):
        raise ValueError(
            f"{source} rule '{rule_id}': kelly_scale = {kelly_scale:.3f} "
            f"out of bounds [{KELLY_SCALE_MIN}, {KELLY_SCALE_MAX}]."
        )

    return RuleEffect(
        delta_fouls=delta_fouls,
        variance_scale=variance_scale,
        kelly_scale=kelly_scale,
    )


def _validate_condition_structure(
    cond: Any,
    rule_id: str,
    source: str,
    depth: int,
) -> None:
    """Recursively validate a DSL condition structure.

    Max nesting depth: 2 (e.g. all > [any > [leaf]]).
    Unknown operators raise ValueError at load time (REQ-5.5).
    """
    if not isinstance(cond, dict):
        raise ValueError(
            f"{source} rule '{rule_id}': condition must be a mapping, got {type(cond).__name__}"
        )

    if depth > 2:
        raise ValueError(
            f"{source} rule '{rule_id}': DSL nesting depth exceeds max of 2"
        )

    # Combinator branches
    if "all" in cond:
        for sub in cond["all"]:
            _validate_condition_structure(sub, rule_id, source, depth + 1)
        return
    if "any" in cond:
        for sub in cond["any"]:
            _validate_condition_structure(sub, rule_id, source, depth + 1)
        return
    if "not" in cond:
        _validate_condition_structure(cond["not"], rule_id, source, depth + 1)
        return

    # Flag shortcuts
    if "flag_present" in cond or "flag_absent" in cond:
        return

    # Field-based leaf
    if "field" in cond:
        ops_in_cond = set(cond.keys()) - {"field"}
        for op in ops_in_cond:
            if op not in VALID_OPERATORS:
                raise ValueError(
                    f"{source} rule '{rule_id}': unknown DSL operator '{op}'. "
                    f"Valid operators: {sorted(VALID_OPERATORS)}"
                )
        if not ops_in_cond:
            raise ValueError(
                f"{source} rule '{rule_id}': field condition has no operator"
            )
        return

    raise ValueError(
        f"{source} rule '{rule_id}': unrecognized condition structure: {cond!r}"
    )
