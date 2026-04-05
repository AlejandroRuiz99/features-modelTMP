"""Interfaz publica del modulo features_generator.

Uso desde scripts externos:

    from generate import generate_features, generate_features_batch

    # Un solo partido -> flat dict listo para el modelo
    feat = generate_features(
        equipo_local="Real Madrid",
        equipo_visitante="Ath Madrid",
        jornada=29,
    )

    # Batch: varios partidos (estado historico cargado una sola vez)
    feats = generate_features_batch([
        {"equipo_local": "Real Madrid", "equipo_visitante": "Ath Madrid", "jornada": 29},
        {"equipo_local": "Barcelona",   "equipo_visitante": "Sevilla",    "jornada": 29},
    ])

Para reports/debug (contrato anidado completo):
    from assembly.feature_assembler import build_detailed
    from core.state_cache import get_state

    contract = build_detailed(state=get_state(), equipo_local_input=..., ...)
"""

from __future__ import annotations

from typing import Any

from core.state_cache import get_state
from assembly import build_features


def _build_kwargs(
    equipo_local: str,
    equipo_visitante: str,
    state: dict[str, Any],
    *,
    jornada: int | None,
    arbitro: str | None,
    fecha_partido: str | None,
    cuotas_prepartido: dict[str, Any] | None,
    features_profile: str | None,
    features_config: str | dict | None,
) -> dict[str, Any]:
    return dict(
        state=state,
        equipo_local_input=equipo_local,
        equipo_visitante_input=equipo_visitante,
        jornada=jornada,
        arbitro_input=arbitro,
        arbitraje_source=("manual" if arbitro else "not_available"),
        cuotas_prepartido=cuotas_prepartido,
        fecha_partido_input=fecha_partido,
        features_profile=features_profile,
        features_config=features_config,
    )


def generate_features(
    equipo_local: str,
    equipo_visitante: str,
    *,
    jornada: int | None = None,
    arbitro: str | None = None,
    fecha_partido: str | None = None,
    cuotas_prepartido: dict[str, Any] | None = None,
    refresh_data: bool = False,
    features_profile: str | None = None,
    features_config: str | dict | None = None,
) -> dict[str, Any]:
    """Genera el flat dict de features para un partido, listo para el modelo.

    El calendario multi-competicion (UCL, Copa, etc.) se obtiene automaticamente
    de la tabla liga_calendar en Supabase (cargada en el state via cal_index).

    Args:
        equipo_local:      Nombre del equipo local (admite alias).
        equipo_visitante:  Nombre del equipo visitante (admite alias).
        jornada:           Jornada de La Liga (opcional, se infiere si None).
        arbitro:           Nombre del arbitro. Si None, se resuelve via API RFEF.
        fecha_partido:     Fecha ISO (YYYY-MM-DD). Mejora resolucion del arbitro
                           y calculo de fatiga/descanso.
        cuotas_prepartido: Dict con cuotas de apuestas ({b365_home, b365_draw, ...}).
        refresh_data:      Fuerza recarga del estado desde Supabase.
        features_profile:  Perfil de features ("prediction", "training", "minimal").
                           Si None, usa el perfil activo en features.yaml.
        features_config:   Ruta a un YAML alternativo o dict directo. Si None, usa
                           features.yaml del directorio de features_generator.
    """
    state = get_state(refresh=refresh_data)
    return build_features(
        **_build_kwargs(
            equipo_local,
            equipo_visitante,
            state,
            jornada=jornada,
            arbitro=arbitro,
            fecha_partido=fecha_partido,
            cuotas_prepartido=cuotas_prepartido,
            features_profile=features_profile,
            features_config=features_config,
        )
    )


def generate_features_batch(
    partidos: list[dict[str, Any]],
    *,
    refresh_data: bool = False,
    features_profile: str | None = None,
    features_config: str | dict | None = None,
) -> list[dict[str, Any]]:
    """Genera flat dict de features para una lista de partidos.

    Carga el estado historico una sola vez y lo reutiliza para todos.
    """
    state = get_state(refresh=refresh_data)
    return [
        build_features(
            **_build_kwargs(
                p["equipo_local"],
                p["equipo_visitante"],
                state,
                jornada=p.get("jornada"),
                arbitro=p.get("arbitro"),
                fecha_partido=p.get("fecha_partido"),
                cuotas_prepartido=p.get("cuotas_prepartido"),
                features_profile=features_profile,
                features_config=features_config,
            )
        )
        for p in partidos
    ]
