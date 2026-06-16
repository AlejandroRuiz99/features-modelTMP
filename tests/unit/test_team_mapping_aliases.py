"""Tests for team_mapping.normalize() covering short-form aliases used by
Codere and other bookmakers in odds_raw.

Regression test for the J30 2025-26 bug where `get_match_odds_rows()` failed
to match 3 of 10 matches because Codere uses short forms like 'Athletic',
'Atlético', and 'Rayo' that were not covered by FBREF_TO_DB.
"""

from __future__ import annotations

import pytest

from selection.team_mapping import (
    BOOKMAKER_SHORT_ALIASES,
    FBREF_TO_DB,
    normalize,
)


class TestNormalizeBookmakerShortAliases:
    """Short-form aliases used by Codere in the odds_raw table."""

    def test_athletic_short_form_maps_to_ath_bilbao(self) -> None:
        """'Athletic' (Codere short form) → 'Ath Bilbao' (canonical)."""
        assert normalize("Athletic") == "Ath Bilbao"

    def test_athletic_case_insensitive(self) -> None:
        """Short aliases are matched case-insensitively."""
        assert normalize("athletic") == "Ath Bilbao"
        assert normalize("ATHLETIC") == "Ath Bilbao"
        assert normalize("AtHlEtIc") == "Ath Bilbao"

    def test_athletic_with_whitespace(self) -> None:
        """Whitespace is stripped before alias lookup."""
        assert normalize("  Athletic  ") == "Ath Bilbao"
        assert normalize("\tAthletic\n") == "Ath Bilbao"

    def test_atletico_short_with_accent_maps_to_ath_madrid(self) -> None:
        """'Atlético' with accent → 'Ath Madrid'."""
        assert normalize("Atlético") == "Ath Madrid"

    def test_atletico_short_without_accent_maps_to_ath_madrid(self) -> None:
        """'Atletico' without accent → 'Ath Madrid'."""
        assert normalize("Atletico") == "Ath Madrid"

    def test_rayo_short_form_maps_to_vallecano(self) -> None:
        """'Rayo' (Codere short form) → 'Vallecano'."""
        assert normalize("Rayo") == "Vallecano"


class TestNormalizeLongFormsStillWork:
    """Regression: the existing FBREF long forms still resolve correctly."""

    def test_athletic_club_still_maps(self) -> None:
        """'Athletic Club' → 'Ath Bilbao' (existing behavior preserved)."""
        assert normalize("Athletic Club") == "Ath Bilbao"

    def test_athletic_bilbao_added(self) -> None:
        """'Athletic Bilbao' → 'Ath Bilbao' (new long form alias)."""
        assert normalize("Athletic Bilbao") == "Ath Bilbao"

    def test_atletico_de_madrid_with_accent(self) -> None:
        """'Atlético de Madrid' → 'Ath Madrid' (existing)."""
        assert normalize("Atlético de Madrid") == "Ath Madrid"

    def test_atletico_de_madrid_without_accent_added(self) -> None:
        """'Atletico de Madrid' without accent → 'Ath Madrid' (new)."""
        assert normalize("Atletico de Madrid") == "Ath Madrid"

    def test_rayo_vallecano_still_maps(self) -> None:
        """'Rayo Vallecano' → 'Vallecano' (existing)."""
        assert normalize("Rayo Vallecano") == "Vallecano"

    def test_real_sociedad_still_maps(self) -> None:
        """'Real Sociedad' → 'Sociedad' (existing)."""
        assert normalize("Real Sociedad") == "Sociedad"

    def test_espanyol_still_maps(self) -> None:
        """'Espanyol' → 'Espanol' (existing)."""
        assert normalize("Espanyol") == "Espanol"

    def test_alaves_still_maps(self) -> None:
        """'Alavés' → 'Alaves' (existing)."""
        assert normalize("Alavés") == "Alaves"


class TestNormalizePassthrough:
    """Names that are already canonical pass through unchanged."""

    def test_barcelona_passthrough(self) -> None:
        assert normalize("Barcelona") == "Barcelona"

    def test_valencia_passthrough(self) -> None:
        assert normalize("Valencia") == "Valencia"

    def test_getafe_passthrough(self) -> None:
        assert normalize("Getafe") == "Getafe"

    def test_unknown_team_passthrough(self) -> None:
        """Names not in any map are returned stripped but unchanged."""
        assert normalize("Unknown FC") == "Unknown FC"
        assert normalize("  Unknown FC  ") == "Unknown FC"


class TestNormalizePrecedence:
    """Short aliases take precedence over the FBREF map."""

    def test_short_alias_wins_over_fbref(self) -> None:
        """If a short alias exists, it is applied BEFORE the FBREF map.

        This matters because 'Athletic' is not in FBREF_TO_DB (only
        'Athletic Club' is), so without the short alias it would fall
        through as-is. The test verifies the precedence ordering in
        the function body.
        """
        # Sanity: 'Athletic' is NOT in FBREF_TO_DB but IS in BOOKMAKER_SHORT_ALIASES
        assert "Athletic" not in FBREF_TO_DB
        assert "athletic" in BOOKMAKER_SHORT_ALIASES
        # And the normalize result matches the short alias value
        assert normalize("Athletic") == BOOKMAKER_SHORT_ALIASES["athletic"]

    def test_short_alias_dict_values_are_canonical(self) -> None:
        """All short alias values must match what FBREF_TO_DB produces."""
        # The canonical values in BOOKMAKER_SHORT_ALIASES must be the same
        # vocabulary as FBREF_TO_DB values, otherwise feature generation
        # breaks.
        fbref_values = set(FBREF_TO_DB.values())
        for short_key, canonical in BOOKMAKER_SHORT_ALIASES.items():
            assert canonical in fbref_values, (
                f"BOOKMAKER_SHORT_ALIASES[{short_key!r}] = {canonical!r} "
                f"is not in FBREF_TO_DB values — canonical vocabulary drift"
            )


class TestJ30CodereRegressionCase:
    """Regression test for the exact J30 2025-26 bug reported.

    In odds_raw, Codere stored these pairs with short forms that did not
    match our partidos.json canonical names without the new aliases:

    - "Athletic vs Getafe"       (should match "Getafe vs Ath Bilbao")
    - "Barcelona vs Atlético"    (should match "Ath Madrid vs Barcelona")
    - "Elche vs Rayo"            (should match "Vallecano vs Elche")
    """

    @pytest.mark.parametrize(
        "codere_raw,expected_canonical",
        [
            ("Athletic", "Ath Bilbao"),
            ("Atlético", "Ath Madrid"),
            ("Rayo", "Vallecano"),
        ],
    )
    def test_codere_short_forms_resolved(
        self,
        codere_raw: str,
        expected_canonical: str,
    ) -> None:
        """All 3 short forms that broke J30 matching are now resolved."""
        assert normalize(codere_raw) == expected_canonical
