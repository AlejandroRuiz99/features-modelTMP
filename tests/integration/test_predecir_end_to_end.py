"""
Integration test skeleton for predecir_jornada end-to-end pipeline.

J29 2025-26 fixture with mocked Playwright/Supabase/subprocess.
Entire module skipped until fixtures are available (T4.1 will remove the skip marker).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="fixtures not yet available — enable after T4.1")
class TestPredecirJornadaEndToEnd:
    """Full pipeline test for Jornada 29 2025-2026."""

    def test_pipeline_j29_non_interactive(self, tmp_path) -> None:
        """
        Full pipeline J29 2025-26: parse → filter → freshness → fbref → batch JSON
        → prediction → odds → EV → markdown.

        Mocks:
        - supabase_client.fetch_matches_for_season → fixture data
        - fbref_calendar.get_schedule_links → synthetic date mapping
        - run_prediction_subprocess → synthetic predictions JSON
        - state_cache / odds_client → synthetic odds rows
        """
        from unittest.mock import patch

        from predecir_jornada import predecir_jornada

        # Synthetic Supabase rows for jornadas 1-28 (all present = no gap)
        supabase_rows = [
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            # ... (abbreviated fixture)
        ]

        synthetic_predictions = [
            {
                "match": "Real Madrid vs Ath Bilbao",
                "home_expected": 13.5,
                "away_expected": 11.8,
                "expected_fouls": 25.3,
                "ou_table": {
                    "21.5": {"p_over": 0.85, "p_under": 0.15},
                    "24.5": {"p_over": 0.55, "p_under": 0.45},
                    "27.5": {"p_over": 0.25, "p_under": 0.75},
                },
            }
        ]

        with (
            patch(
                "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
            ),
            patch("predecir_jornada.get_schedule_links", return_value={}),
            patch(
                "predecir_jornada.run_prediction_subprocess",
                return_value=synthetic_predictions,
            ),
            patch("predecir_jornada.fetch_codere_odds", return_value="no_disponibles"),
        ):
            result = predecir_jornada(
                jornada=29,
                temporada="2025-2026",
                interactive=False,
                no_fbref=True,
                no_update_stats=False,
            )

        assert result is not None
        assert "predictions" in result
        assert "markdown" in result

    def test_pipeline_no_odds_renders_clean(self, tmp_path) -> None:
        """Future matchday w/o odds → 'no_disponibles' rendered cleanly."""
        from unittest.mock import patch

        from predecir_jornada import predecir_jornada

        with (
            patch("predecir_jornada.fetch_matches_for_season", return_value=[]),
            patch("predecir_jornada.get_schedule_links", return_value={}),
            patch("predecir_jornada.run_prediction_subprocess", return_value=[]),
            patch("predecir_jornada.fetch_codere_odds", return_value="no_disponibles"),
        ):
            result = predecir_jornada(
                jornada=35,
                temporada="2025-2026",
                interactive=False,
                no_fbref=True,
                no_update_stats=True,
            )
        # Should not crash
        assert isinstance(result, dict)
