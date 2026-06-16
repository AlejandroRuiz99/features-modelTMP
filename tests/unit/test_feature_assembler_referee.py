"""Unit tests for feature_assembler.py referee-related field propagation.

Covers:
  - _flatten: referee_is_shrunk key included in the flat feature dict (D1, T1.7/T1.8)
"""

from __future__ import annotations

# features_generator is on sys.path via conftest.py (pythonpath in pyproject.toml)
from assembly.feature_assembler import _flatten

# ---------------------------------------------------------------------------
# Minimal synthetic nested contract dict for _flatten
# ---------------------------------------------------------------------------


def _make_raw_contract(
    *,
    arb_is_shrunk: bool = True,
    arb_n_partidos: int = 4,
    arb_nombre: str = "TestRef",
) -> dict:
    """
    Build a minimal nested contract dict (as produced by _assemble_contract)
    suitable for passing to _flatten.

    Only the fields read by _flatten are populated; all others use sensible
    defaults so the function does not crash.
    """
    arb_estadisticas = {
        "mu_permisivo": 22.0,
        "mu_estricto": 30.0,
        "sigma_permisivo": 4.0,
        "sigma_estricto": 4.0,
        "peso_estricto": 0.5,
        "partidos_arbitrados": arb_n_partidos,
        "is_shrunk": arb_is_shrunk,
        "fouls_clean_avg": 25.0,
    }
    return {
        "partido": {
            "equipo_local": "Home FC",
            "equipo_visitante": "Away FC",
            "jornada": 20,
            "temporada": "2025-26",
            "fecha": "2026-05-02",
        },
        "arbitro": {
            "nombre": arb_nombre,
            "estadisticas": arb_estadisticas,
            "interaccion_local": {"delta_faltas": 0.0, "partidos": 0},
            "interaccion_visitante": {"delta_faltas": 0.0, "partidos": 0},
            "home_bias": 0.5,
        },
        "equipos": {
            "local": {
                "temporada_completa": {
                    "faltas_cometidas": 12.0,
                    "faltas_provocadas": 12.0,
                    "tiros": 12.0,
                    "corners": 5.0,
                    "amarillas": 2.0,
                    "rojas": 0.1,
                },
                "forma_reciente": {"faltas_media": 12.0, "momentum": 0.5},
                "contexto": {
                    "urgencia": 0.5,
                    "fatiga": 0.2,
                    "dias_descanso": 7,
                    "factor_xfaltas": 1.0,
                },
                "clasificacion": {"posicion": 10},
            },
            "visitante": {
                "temporada_completa": {
                    "faltas_cometidas": 12.0,
                    "faltas_provocadas": 12.0,
                    "tiros": 12.0,
                    "corners": 5.0,
                    "amarillas": 2.0,
                    "rojas": 0.1,
                },
                "forma_reciente": {"faltas_media": 12.0, "momentum": 0.5},
                "contexto": {
                    "urgencia": 0.5,
                    "fatiga": 0.2,
                    "dias_descanso": 7,
                    "factor_xfaltas": 1.0,
                },
                "clasificacion": {"posicion": 15},
            },
        },
        "metricas_esperadas": {
            "xgoles": {"local": 1.2, "visitante": 0.9},
            "xfaltas": {"local": 12.5, "visitante": 12.5},
            "xposesion": {"local": 50.0, "visitante": 50.0},
            "agresividad": {"local": 0.5, "visitante": 0.5, "total": 0.5},
            "volumen": {"pace_index": 31.0},
        },
        "mercado": {
            "resultado": {
                "prob_local": 0.40,
                "prob_empate": 0.25,
                "prob_visitante": 0.35,
            },
            "goles_ou": {"prob_over": 0.50, "prob_under": 0.50},
            "faltas_total_ou": {"prob_over": 0.50, "prob_under": 0.50, "linea": 24.5},
            "derivadas": {
                "has_market_odds": False,
                "market_favorite_prob": 0.40,
                "market_balance": 1.0,
                "market_entropy": 1.0,
            },
        },
        "contexto_partido": {
            "intensidad_esperada": "media",
            "riesgo_disciplinario": "medio",
            "h2h_faltas_media": 25.0,
            "h2h_partidos": 5,
            "season_phase": 0.5,
            "is_derby": False,
        },
    }


# ---------------------------------------------------------------------------
# T1.7 — _flatten includes referee_is_shrunk from arb_stats["is_shrunk"]
# ---------------------------------------------------------------------------


class TestFlattenRefereeIsShrunk:
    """D1: _flatten must include referee_is_shrunk taken from arb_stats['is_shrunk']."""

    def test_referee_is_shrunk_key_present_in_flat_dict(self) -> None:
        """The flat dict produced by _flatten must contain the key 'referee_is_shrunk'."""
        raw = _make_raw_contract(arb_is_shrunk=True, arb_n_partidos=4)
        flat = _flatten(raw)
        assert "referee_is_shrunk" in flat, (
            f"Expected 'referee_is_shrunk' in flat dict keys. Got: {sorted(flat.keys())}"
        )

    def test_referee_is_shrunk_true_propagated(self) -> None:
        """arb_stats['is_shrunk']=True → flat dict has referee_is_shrunk=True."""
        raw = _make_raw_contract(arb_is_shrunk=True, arb_n_partidos=4)
        flat = _flatten(raw)
        assert flat["referee_is_shrunk"] is True, (
            f"Expected referee_is_shrunk=True, got {flat['referee_is_shrunk']}"
        )

    def test_referee_is_shrunk_false_propagated(self) -> None:
        """arb_stats['is_shrunk']=False → flat dict has referee_is_shrunk=False."""
        raw = _make_raw_contract(arb_is_shrunk=False, arb_n_partidos=15)
        flat = _flatten(raw)
        assert flat["referee_is_shrunk"] is False, (
            f"Expected referee_is_shrunk=False, got {flat['referee_is_shrunk']}"
        )

    def test_referee_is_shrunk_absent_defaults_false(self) -> None:
        """If 'is_shrunk' absent from arb_stats, referee_is_shrunk defaults to False."""
        raw = _make_raw_contract(arb_is_shrunk=True, arb_n_partidos=4)
        # Remove is_shrunk from the estadisticas to simulate an old contract
        del raw["arbitro"]["estadisticas"]["is_shrunk"]
        flat = _flatten(raw)
        # Key must still be present, defaulting to False
        assert "referee_is_shrunk" in flat, (
            "Expected 'referee_is_shrunk' key even when arb_stats lacks 'is_shrunk'"
        )
        assert flat["referee_is_shrunk"] is False, (
            f"Expected default False when key absent, got {flat['referee_is_shrunk']}"
        )
