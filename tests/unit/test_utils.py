"""Unit tests for core/utils.py — pure utility functions."""

from __future__ import annotations

import math
import pytest

from core.utils import (
    fuzzy_name_search,
    no_vig,
    no_vig_3,
    market_entropy,
    norm_text,
    team_match,
    extract_ou,
    safe_float,
    parse_date_safe,
    classify_competition,
    dedupe_dicts,
    TEAM_ALIASES,
)


class TestFuzzyNameSearch:
    """Tests for fuzzy_name_search function."""

    def test_exact_alias_match(self) -> None:
        """Alias exacto devuelve el nombre canónico."""
        candidates = ["Barcelona", "Real Madrid", "Ath Bilbao"]
        assert fuzzy_name_search("barca", candidates, TEAM_ALIASES) == "Barcelona"

    def test_exact_normalized_match(self) -> None:
        """Nombre normalizado exacto devuelve el candidato."""
        candidates = ["Barcelona", "Real Madrid", "Ath Bilbao"]
        assert fuzzy_name_search("BARCELONA", candidates) == "Barcelona"

    def test_substring_query_in_candidate(self) -> None:
        """Substring query en candidato devuelve match más corto."""
        candidates = ["Real Madrid", "Real Sociedad", "Real Betis"]
        result = fuzzy_name_search("real", candidates)
        assert result in candidates

    def test_substring_candidate_in_query(self) -> None:
        """Substring candidato en query devuelve match más largo."""
        candidates = ["Girona", "Las Palmas"]
        result = fuzzy_name_search("ud las palmas", candidates)
        assert result == "Las Palmas"

    def test_fuzzy_alias_match(self) -> None:
        """Fuzzy match con alias devuelve el correcto."""
        candidates = ["Ath Bilbao", "Ath Madrid"]
        result = fuzzy_name_search("bilbao athletic", candidates, TEAM_ALIASES)
        assert result == "Ath Bilbao"

    def test_empty_query_returns_none(self) -> None:
        """Query vacío devuelve None."""
        candidates = ["Barcelona"]
        assert fuzzy_name_search("", candidates) is None
        assert fuzzy_name_search(None, candidates) is None  # type: ignore

    def test_no_match_returns_none(self) -> None:
        """Sin match devuelve None."""
        candidates = ["Barcelona", "Real Madrid"]
        assert fuzzy_name_search("xyz123notreal", candidates) is None


class TestNoVig:
    """Tests for no_vig function (margin removal for 2 outcomes)."""

    def test_standard_odds(self) -> None:
        """Cuotas estándar eliminan margen correctamente."""
        # Cuota 2.0 vs 2.0 deberían dar ~50/50
        p_a, p_b = no_vig(2.0, 2.0)
        assert abs(p_a - 0.5) < 0.01
        assert abs(p_b - 0.5) < 0.01

    def test_asymmetric_odds(self) -> None:
        """Cuotas asimétricas."""
        # Favorito claro: 1.5 vs 6.0
        p_a, p_b = no_vig(1.5, 6.0)
        assert p_a > p_b
        assert 0 < p_a < 1
        assert 0 < p_b < 1

    def test_invalid_odds_returns_default(self) -> None:
        """Cuotas <= 1.01 devuelven 0.5/0.5."""
        assert no_vig(1.0, 2.0) == (0.5, 0.5)
        assert no_vig(2.0, 1.0) == (0.5, 0.5)


class TestNoVig3:
    """Tests for no_vig_3 function (margin removal for 1X2)."""

    def test_standard_1x2(self) -> None:
        """Cuotas 1X2 estándar."""
        p1, px, p2 = no_vig_3(2.5, 3.3, 2.8)
        assert abs(p1 + px + p2 - 1.0) < 0.01
        assert all(0 < p < 1 for p in (p1, px, p2))

    def test_invalid_returns_default(self) -> None:
        """Cuotas inválidas devuelven default."""
        p1, px, p2 = no_vig_3(1.0, 3.0, 4.0)
        assert p1 == pytest.approx(0.40)
        assert px == pytest.approx(0.25)
        assert p2 == pytest.approx(0.35)


class TestMarketEntropy:
    """Tests for market_entropy function."""

    def test_uniform_distribution(self) -> None:
        """Distribución uniforme tiene máxima entropía."""
        # Para 3 outcomes, max entropy = ln(3) ≈ 1.099
        e = market_entropy(1 / 3, 1 / 3, 1 / 3)
        assert abs(e - math.log(3)) < 0.01

    def test_deterministic_distribution(self) -> None:
        """Distribución determinista tiene entropía 0."""
        assert market_entropy(1.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_zero_probability_handled(self) -> None:
        """Probabilidades cero no causan log(0)."""
        # Debería usar 1e-12 floor
        e = market_entropy(0.0, 0.5, 0.5)
        assert e > 0  # Puede calcular entropy


class TestNormText:
    """Tests for norm_text function."""

    def test_normalizes_accents(self) -> None:
        """Accents se normalizan a ASCII."""
        assert norm_text("Málaga") == "malaga"
        assert norm_text("Alavés") == "alaves"

    def test_lowercases(self) -> None:
        """String se convierte a lowercase."""
        assert norm_text("BARCELONA") == "barcelona"

    def test_strips_whitespace(self) -> None:
        """Whitespace se elimina."""
        assert norm_text("  real madrid  ") == "real madrid"

    def test_collapses_multiple_spaces(self) -> None:
        """Múltiples espacios se colapsan a uno."""
        assert norm_text("real   madrid") == "real madrid"

    def test_empty_returns_empty(self) -> None:
        """String vacío o None devuelve empty string."""
        assert norm_text("") == ""
        assert norm_text(None) == ""


class TestTeamMatch:
    """Tests for team_match function."""

    def test_exact_match(self) -> None:
        """Nombres exactos hacen match."""
        assert team_match("Barcelona", "barcelona") is True

    def test_substring_match(self) -> None:
        """Substring hace match."""
        assert team_match("Barcelona FC", "Barcelona") is True
        assert team_match("Barcelona", "Barcelona FC") is True

    def test_fuzzy_match_high_similarity(self) -> None:
        """Nombres similares con ratio >= 0.82 hacen match."""
        # "Real Madrid" vs "RealMadrid" ratio ~0.96
        assert team_match("RealMadrid", "Real Madrid") is True

    def test_no_match_low_similarity(self) -> None:
        """Nombres diferentes no hacen match."""
        assert team_match("Barcelona", "Madrid") is False

    def test_empty_returns_false(self) -> None:
        """Empty/null devuelve False."""
        assert team_match(None, "Barcelona") is False
        assert team_match("Barcelona", None) is False


class TestExtractOu:
    """Tests for extract_ou function (Over/Under extraction)."""

    def test_over_spanish(self) -> None:
        """Extrae Over en español."""
        side, line = extract_ou("Más de 2.5 goles")
        assert side == "over"
        assert line == 2.5

    def test_under_english(self) -> None:
        """Extrae Under en inglés."""
        side, line = extract_ou("Under 3.5 corners")
        assert side == "under"
        assert line == 3.5

    def test_comma_decimal(self) -> None:
        """Decimal con coma se parsea."""
        side, line = extract_ou("Más de 2,5")
        assert line == 2.5

    def test_no_line(self) -> None:
        """Sin línea devuelve None para el float."""
        side, line = extract_ou("over goles")
        assert side == "over"
        assert line is None

    def test_no_side(self) -> None:
        """Sin over/under devuelve None para el side."""
        side, line = extract_ou("2.5 goles")
        assert side is None
        assert line == 2.5

    def test_empty_returns_none_none(self) -> None:
        """Empty devuelve (None, None)."""
        assert extract_ou("") == (None, None)
        assert extract_ou(None) == (None, None)


class TestSafeFloat:
    """Tests for safe_float function."""

    def test_valid_float(self) -> None:
        """Float válido se devuelve."""
        assert safe_float(3.14) == 3.14
        assert safe_float("2.5") == 2.5

    def test_invalid_returns_none(self) -> None:
        """Valor inválido devuelve None."""
        assert safe_float("abc") is None
        assert safe_float(None) is None

    def test_value_at_threshold_returns_none(self) -> None:
        """Valor <= 1.0 devuelve None (threshold for odds)."""
        assert safe_float(0.5) is None
        assert safe_float(1.0) is None
        assert safe_float(0.999) is None

    def test_valid_value_above_one(self) -> None:
        """Valor > 1.0 se devuelve."""
        assert safe_float(1.009) == 1.009  # Just above threshold
        assert safe_float(1.01) == 1.01
        assert safe_float(2.5) == 2.5


class TestParseDateSafe:
    """Tests for parse_date_safe function."""

    def test_valid_date(self) -> None:
        """Fecha válida se parsea."""
        from datetime import date

        result = parse_date_safe("2026-03-31")
        assert result == date(2026, 3, 31)

    def test_datetime_string_truncated(self) -> None:
        """Timestamp largo se trunca a fecha."""
        from datetime import date

        result = parse_date_safe("2026-03-31 14:30:00")
        assert result == date(2026, 3, 31)

    def test_invalid_returns_none(self) -> None:
        """Formato inválido devuelve None."""
        assert parse_date_safe("31/03/2026") is None
        assert parse_date_safe("not a date") is None


class TestClassifyCompetition:
    """Tests for classify_competition function."""

    def test_champions(self) -> None:
        """Champions se clasifica como ucl."""
        assert classify_competition("Champions League") == "ucl"
        assert classify_competition("UCL Final") == "ucl"

    def test_europa(self) -> None:
        """Europa League se clasifica como uel."""
        assert classify_competition("Europa League") == "uel"
        assert classify_competition("UEL") == "uel"

    def test_conference(self) -> None:
        """Conference League se clasifica como uecl."""
        assert classify_competition("Europa Conference League") == "uecl"
        assert classify_competition("UECL") == "uecl"

    def test_copa(self) -> None:
        """Copa del Rey se clasifica como copa."""
        assert classify_competition("Copa del Rey") == "copa"
        assert classify_competition("Spanish Cup") == "copa"

    def test_laliga(self) -> None:
        """La Liga se clasifica como liga."""
        assert classify_competition("LaLiga") == "liga"
        assert classify_competition("La Liga EA Sports") == "liga"

    def test_unknown(self) -> None:
        """None o desconocido devuelve 'desconocida' u 'otra'."""
        assert classify_competition(None) == "desconocida"
        assert classify_competition("") == "desconocida"
        assert classify_competition("Friendly") == "otra"


class TestDedupeDicts:
    """Tests for dedupe_dicts function."""

    def test_removes_duplicates_by_key(self) -> None:
        """Elimina duplicados por keys especificadas."""
        items = [
            {"id": 1, "name": "A"},
            {"id": 1, "name": "B"},  # duplicate id
            {"id": 2, "name": "C"},
        ]
        result = dedupe_dicts(items, ["id"])
        assert len(result) == 2
        assert result[0]["name"] == "A"  # first wins

    def test_multiple_keys(self) -> None:
        """Dedupe por múltiples keys."""
        items = [
            {"match": 1, "team": "A", "val": 10},
            {"match": 1, "team": "A", "val": 20},  # duplicate
            {"match": 1, "team": "B", "val": 30},
        ]
        result = dedupe_dicts(items, ["match", "team"])
        assert len(result) == 2

    def test_empty_returns_empty(self) -> None:
        """Lista vacía devuelve vacía."""
        assert dedupe_dicts([], ["id"]) == []

    def test_preserves_order(self) -> None:
        """Preserva orden de primera aparición."""
        items = [
            {"id": "a", "x": 1},
            {"id": "b", "x": 2},
            {"id": "a", "x": 3},
        ]
        result = dedupe_dicts(items, ["id"])
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"
