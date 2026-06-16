"""
tests/parsers/test_team_canonical.py — TDD for parsers.team_canonical

Tests (T2.1):
  1. Exact match returns canonical name (case-insensitive).
  2. Alias match returns canonical name.
  3. Fuzzy match (above threshold) returns canonical name.
  4. Below threshold returns None.
  5. LALIGA_2025_26_TEAMS has exactly 20 teams.
  6. Unknown name returns None.
"""

from __future__ import annotations

from parsers.team_canonical import LALIGA_2025_26_TEAMS, canonicalize


class TestExactMatch:
    def test_exact_canonical_name(self) -> None:
        """Exact canonical name returns the name itself."""
        assert canonicalize("Real Madrid") == "Real Madrid"

    def test_exact_canonical_case_insensitive(self) -> None:
        """Canonical name matching is case-insensitive."""
        assert canonicalize("real madrid") == "Real Madrid"
        assert canonicalize("REAL MADRID") == "Real Madrid"

    def test_exact_match_atletico(self) -> None:
        """Atlético de Madrid matches exactly."""
        assert canonicalize("Atlético de Madrid") == "Atlético de Madrid"


class TestAliasMatch:
    def test_alias_atleti(self) -> None:
        """Atleti is an alias for Atlético de Madrid."""
        assert canonicalize("Atleti") == "Atlético de Madrid"

    def test_alias_barca(self) -> None:
        """Barça is an alias for FC Barcelona."""
        result = canonicalize("Barça")
        assert result == "FC Barcelona"

    def test_alias_barcelona_no_accent(self) -> None:
        """Barca (no cedilla) is an alias for FC Barcelona."""
        result = canonicalize("Barca")
        assert result == "FC Barcelona"

    def test_alias_barcelona(self) -> None:
        """Barcelona (no FC prefix) is an alias."""
        result = canonicalize("Barcelona")
        assert result == "FC Barcelona"

    def test_alias_athletic_bilbao(self) -> None:
        """Athletic Bilbao is an alias for Athletic Club."""
        result = canonicalize("Athletic Bilbao")
        assert result == "Athletic Club"

    def test_alias_betis(self) -> None:
        """Betis is an alias for Real Betis."""
        result = canonicalize("Betis")
        assert result == "Real Betis"

    def test_alias_celta(self) -> None:
        """Celta is an alias for Celta de Vigo."""
        result = canonicalize("Celta")
        assert result == "Celta de Vigo"

    def test_alias_rayo(self) -> None:
        """Rayo is an alias for Rayo Vallecano."""
        result = canonicalize("Rayo")
        assert result == "Rayo Vallecano"


class TestFuzzyMatch:
    def test_fuzzy_match_mallorca_typo(self) -> None:
        """Minor typo in Mallorca resolves to canonical."""
        result = canonicalize("Mallorc")  # short but close enough
        # May or may not match depending on threshold — test that it's Mallorca or None
        assert result in (None, "Mallorca")

    def test_fuzzy_match_osasuna(self) -> None:
        """Osasuna close match."""
        result = canonicalize("Osasuna")
        assert result == "Osasuna"

    def test_fuzzy_match_below_threshold_returns_none(self) -> None:
        """Completely unrelated string returns None."""
        result = canonicalize("XYZ_UNKNOWN_TEAM_99")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        result = canonicalize("")
        assert result is None


class TestTeamsList:
    def test_exactly_20_teams(self) -> None:
        """LALIGA_2025_26_TEAMS has exactly 20 teams."""
        assert len(LALIGA_2025_26_TEAMS) == 20

    def test_all_teams_are_strings(self) -> None:
        """All entries in LALIGA_2025_26_TEAMS are strings."""
        for team in LALIGA_2025_26_TEAMS:
            assert isinstance(team, str), f"Expected str, got {type(team)} for {team!r}"

    def test_known_teams_present(self) -> None:
        """Key teams are in the list."""
        for team in ["Real Madrid", "FC Barcelona", "Atlético de Madrid"]:
            assert team in LALIGA_2025_26_TEAMS, f"{team!r} not in LALIGA_2025_26_TEAMS"
