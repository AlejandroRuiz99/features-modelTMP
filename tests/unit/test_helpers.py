"""Unit tests for core/helpers.py — pure helper functions."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from core.helpers import parse_date, decay_weight, safe, clip


class TestParseDate:
    """Tests for parse_date function."""

    def test_valid_iso_date(self) -> None:
        """Fecha ISO válida se parsea."""
        result = parse_date("2026-03-31")
        assert result == date(2026, 3, 31)

    def test_different_dates(self) -> None:
        """Diferentes fechas se parsean correctamente."""
        cases = [
            ("2024-01-01", date(2024, 1, 1)),
            ("2025-12-31", date(2025, 12, 31)),
            ("2020-02-29", date(2020, 2, 29)),  # Leap year
        ]
        for input_str, expected in cases:
            assert parse_date(input_str) == expected

    def test_invalid_format_raises(self) -> None:
        """Formato inválido lanza ValueError."""
        with pytest.raises(ValueError):
            parse_date("31/03/2026")  # Wrong format

        with pytest.raises(ValueError):
            parse_date("not a date")  # Invalid string

        with pytest.raises(ValueError):
            parse_date("2026-13-01")  # Invalid month


class TestDecayWeight:
    """Tests for decay_weight function."""

    def test_same_day_weight_is_one(self) -> None:
        """Mismo día tiene peso 1."""
        reference = date(2026, 3, 31)
        result = decay_weight(reference, reference, lam=0.05)
        assert result == pytest.approx(1.0)

    def test_exponential_decay(self) -> None:
        """Decaimiento exponencial con dias."""
        reference = date(2026, 3, 31)
        # 100 días de diferencia con lambda=0.05
        past = reference - timedelta(days=100)
        result = decay_weight(past, reference, lam=0.05)
        expected = math.exp(-0.05 * 100)
        assert result == pytest.approx(expected)

    def test_recent_high_weight(self) -> None:
        """Partidos recientes tienen peso alto."""
        reference = date(2026, 3, 31)
        recent = reference - timedelta(days=7)
        old = reference - timedelta(days=365)
        w_recent = decay_weight(recent, reference, lam=0.02)
        w_old = decay_weight(old, reference, lam=0.02)
        assert w_recent > w_old

    def test_lambda_affects_decay_rate(self) -> None:
        """Lambda más alto = decaimiento más rápido."""
        reference = date(2026, 3, 31)
        past = reference - timedelta(days=100)

        w_slow = decay_weight(past, reference, lam=0.01)
        w_fast = decay_weight(past, reference, lam=0.10)

        assert w_slow > w_fast

    def test_zero_days_zero_weight_at_limit(self) -> None:
        """Días cero dan peso 1, días muy lejanos ~ 0."""
        reference = date(2026, 3, 31)

        # 0 days
        assert decay_weight(reference, reference, lam=0.05) == pytest.approx(1.0)

        # Many days (extreme case)
        very_old = reference - timedelta(days=1000)
        w = decay_weight(very_old, reference, lam=0.05)
        assert w < 1e-20  # Essentially zero


class TestSafe:
    """Tests for safe function."""

    def test_returns_value_when_not_none(self) -> None:
        """Devuelve el valor si no es None."""
        assert safe(42) == 42
        assert safe("hello") == "hello"
        assert safe(3.14) == 3.14

    def test_returns_default_when_none(self) -> None:
        """Devuelve el default si valor es None."""
        assert safe(None, default=0) == 0
        assert safe(None, default="default") == "default"
        assert safe(None, default=99) == 99

    def test_default_is_zero_by_default(self) -> None:
        """Default por defecto es 0."""
        assert safe(None) == 0

    def test_falsy_values_not_treated_as_none(self) -> None:
        """Falsy values (0, '', False) no son None."""
        assert safe(0) == 0
        assert safe("") == ""
        assert safe(False) is False
        assert safe([], default=["x"]) == []

    def test_generic_type_preservation_int(self) -> None:
        """Preserva el tipo int cuando valor no es None."""
        result: int = safe(42)
        assert result == 42
        assert isinstance(result, int)

    def test_generic_type_preservation_str(self) -> None:
        """Preserva el tipo str cuando valor no es None."""
        result: str = safe("test", default="")
        assert result == "test"
        assert isinstance(result, str)

    def test_generic_default_type_matches(self) -> None:
        """El default tiene el mismo tipo que el valor esperado."""
        result: float = safe(None, default=3.14)
        assert result == 3.14
        assert isinstance(result, float)


class TestClip:
    """Tests for clip function."""

    def test_value_in_range_unchanged(self) -> None:
        """Valor en rango se devuelve sin cambio."""
        assert clip(5.0, 0.0, 10.0) == 5.0
        assert clip(0.0, 0.0, 10.0) == 0.0
        assert clip(10.0, 0.0, 10.0) == 10.0

    def test_value_below_lo_clipped_to_lo(self) -> None:
        """Valor menor que lo se trunca a lo."""
        assert clip(-5.0, 0.0, 10.0) == 0.0
        assert clip(0.5, 1.0, 5.0) == 1.0

    def test_value_above_hi_clipped_to_hi(self) -> None:
        """Valor mayor que hi se trunca a hi."""
        assert clip(15.0, 0.0, 10.0) == 10.0
        assert clip(100.0, 0.0, 1.0) == 1.0

    def test_negative_range(self) -> None:
        """Rango negativo funciona correctamente."""
        assert clip(-5.0, -10.0, 0.0) == -5.0
        assert clip(-15.0, -10.0, 0.0) == -10.0
        assert clip(5.0, -10.0, 0.0) == 0.0

    def test_equal_bounds(self) -> None:
        """Si lo == hi, siempre devuelve ese valor."""
        assert clip(5.0, 3.0, 3.0) == 3.0
        assert clip(100.0, 3.0, 3.0) == 3.0
