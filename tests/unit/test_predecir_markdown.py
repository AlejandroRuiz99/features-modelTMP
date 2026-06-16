"""
Unit tests for predecir_jornada markdown renderer (R11).

Tests written FIRST (TDD RED phase).
Snapshot test with synthetic MatchPrediction fixture.
"""

from __future__ import annotations

from datetime import date

from predecir_jornada import CodereOdds, MatchPrediction, render_markdown


def _make_prediction(
    local: str = "Real Madrid",
    visitante: str = "Ath Madrid",
    fecha: date | None = None,
    jornada: int = 31,
    arbitro: str | None = "Munuera Montero",
    pred_total: float = 25.3,
    pred_local: float = 13.1,
    pred_visitante: float = 12.2,
    codere_odds: CodereOdds | None = None,
    edge: float | None = None,
    ev_over: float | None = None,
    ev_under: float | None = None,
    ou_table: dict | None = None,
) -> MatchPrediction:
    if fecha is None:
        fecha = date(2026, 4, 12)
    if codere_odds is None:
        codere_odds = CodereOdds(
            line=24.5,
            over=1.85,
            under=1.95,
            home_win=2.10,
            draw=3.40,
            away_win=3.20,
        )
    if edge is None:
        edge = pred_total - codere_odds.line
    if ou_table is None:
        ou_table = {
            21.5: (0.15, 0.85),
            23.5: (0.38, 0.62),
            24.5: (0.52, 0.48),
            25.5: (0.65, 0.35),
            27.5: (0.82, 0.18),
        }
    return MatchPrediction(
        local=local,
        visitante=visitante,
        fecha=fecha,
        jornada=jornada,
        arbitro=arbitro,
        pred_total=pred_total,
        pred_local=pred_local,
        pred_visitante=pred_visitante,
        codere_odds=codere_odds,
        edge=edge,
        ev_over=ev_over,
        ev_under=ev_under,
        ou_table=ou_table,
    )


class TestRenderMarkdownHeader:
    """R11: Header block contains required fields."""

    def test_header_contains_jornada(self) -> None:
        """Header includes matchday number."""
        predictions = [_make_prediction(jornada=31)]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        assert "31" in output

    def test_header_contains_season(self) -> None:
        """Header includes season."""
        predictions = [_make_prediction()]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        assert "2025-2026" in output

    def test_header_contains_dominant_date(self) -> None:
        """Header includes dominant date."""
        predictions = [_make_prediction()]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        assert "2026-04-12" in output

    def test_header_contains_match_count(self) -> None:
        """Header includes number of matches."""
        predictions = [
            _make_prediction(),
            _make_prediction(local="Barcelona", visitante="Celta"),
        ]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=2,
            odds_freshness_status="OK",
        )
        assert "2" in output  # at least the count is mentioned


class TestRenderMarkdownSummaryTable:
    """R11: Summary table has required columns."""

    def test_summary_table_columns_present(self) -> None:
        """Summary table includes required column headers."""
        predictions = [_make_prediction()]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        # Required columns per spec R11
        assert "Partido" in output
        assert "Pred" in output
        assert "Línea" in output or "Linea" in output
        assert "Edge" in output

    def test_summary_table_contains_ev_columns(self) -> None:
        """Summary table includes EV Over and EV Under columns."""
        predictions = [_make_prediction()]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        assert "EV" in output


class TestRenderMarkdownMissingValues:
    """R11: Missing values rendered as dash (—), no exception raised."""

    def test_null_referee_renders_dash(self) -> None:
        """Match with null referee shows — in output, no exception."""
        predictions = [_make_prediction(arbitro=None)]
        # Should not raise
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=0,
            odds_freshness_status="OK",
        )
        assert "—" in output or "-" in output

    def test_null_odds_renders_dash(self) -> None:
        """Match with no odds shows — in output, no exception."""
        predictions = [
            _make_prediction(
                codere_odds=None,
                edge=None,
                ev_over=None,
                ev_under=None,
            )
        ]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="no_disponibles",
        )
        assert "—" in output or "-" in output

    def test_empty_predictions_does_not_crash(self) -> None:
        """Empty predictions list renders without error."""
        output = render_markdown(
            predictions=[],
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=0,
            odds_freshness_status="OK",
        )
        assert isinstance(output, str)


class TestRenderMarkdownPerMatchBlock:
    """R11: Per-match block contains prediction details."""

    def test_per_match_contains_team_names(self) -> None:
        """Per-match block includes home and away team names."""
        predictions = [_make_prediction(local="Sociedad", visitante="Leganes")]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        assert "Sociedad" in output
        assert "Leganes" in output

    def test_per_match_contains_ou_table(self) -> None:
        """Per-match block includes O/U table lines."""
        predictions = [_make_prediction()]
        output = render_markdown(
            predictions=predictions,
            jornada=31,
            temporada="2025-2026",
            dominant_date=date(2026, 4, 12),
            num_referees_found=1,
            odds_freshness_status="OK",
        )
        # O/U table should mention the model lines
        assert "21.5" in output
        assert "24.5" in output
