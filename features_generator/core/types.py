"""Tipos estructurados para el pipeline de features.

Type hints para los dicts principales:
  - State: resultado de build_state()
  - FeatureContract: resultado de _assemble_contract()

Estos tipos mejoran el autocompletado y permiten validación estática
sin afectar el runtime (TypedDict es solo para type checking).
"""

from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Tipos anidados para State
# ---------------------------------------------------------------------------


class TeamScore(TypedDict):
    """Estadísticas agregadas de un equipo desde scores."""

    avg_fouls_committed: float
    avg_fouls_suffered: float
    avg_yellow_cards: float
    avg_red_cards: float
    matches_analyzed: int


class RefProfile(TypedDict):
    """Perfil estadístico de un árbitro."""

    factor_fouls: float
    factor_cards: float
    matches: int
    fouls_per_match: float
    cards_per_match: float


class RefGMMParams(TypedDict):
    """Parámetros del modelo GMM de árbitro."""

    mu: float
    sigma: float
    weight: float


class CalendarEntry(TypedDict):
    """Entrada del calendario para un equipo."""

    date: str
    competition: str
    status: str
    is_home: bool
    opponent: str


class State(TypedDict):
    """Estado precalculado para generación de features.

    Resultado de build_state() — contiene todos los datos caches
    necesarios para generar features de un partido.
    """

    partidos: list[dict]
    scores: dict[str, TeamScore]
    xstyles: dict[str, dict]
    ref_perfiles: dict[str, RefProfile]
    perfiles_gmm: dict[str, RefGMMParams]
    objectives: dict[str, dict]
    cal_index: dict[str, list[CalendarEntry]]
    updated_at: str


# ---------------------------------------------------------------------------
# Tipos anidados para FeatureContract
# ---------------------------------------------------------------------------


class MatchInfo(TypedDict, total=False):
    """Metadatos del partido."""

    equipo_local: str
    equipo_visitante: str
    jornada: int
    temporada: str
    fecha: str


class RefereeStats(TypedDict, total=False):
    """Estadísticas del árbitro."""

    nombre: str
    factor_fouls: float
    factor_cards: float
    fouls_per_match: float
    cards_per_match: float


class RefereeInteraction(TypedDict):
    """Interacción histórica árbitro-equipo."""

    delta_faltas: float
    partidos: int


class TeamClassification(TypedDict, total=False):
    """Datos de clasificación del equipo."""

    posicion: int
    puntos: int
    partidos_jugados: int
    diferencia_goles: int


class TeamSeasonStats(TypedDict, total=False):
    """Estadísticas de temporada completa."""

    faltas_cometidas: float
    faltas_provocadas: float
    tiros: float
    corners: float
    posesion: float
    amarillas: float
    rojas: float


class TeamForm(TypedDict, total=False):
    """Forma reciente del equipo."""

    partidos: int
    faltas_media: float
    tarjetas_media: float
    momentum: float


class TeamContext(TypedDict, total=False):
    """Contexto competitivo del equipo."""

    urgencia: float
    fatiga: float
    dias_descanso: int
    factor_xfaltas: float


class TeamBlock(TypedDict, total=False):
    """Bloque completo de un equipo en el contrato."""

    clasificacion: TeamClassification
    temporada_completa: TeamSeasonStats
    forma_reciente: TeamForm
    contexto: TeamContext


class XGoals(TypedDict, total=False):
    """Goles esperados."""

    local: float
    visitante: float
    total: float


class XFouls(TypedDict, total=False):
    """Faltas esperadas."""

    local: float
    visitante: float
    total: float
    media_liga: float


class XPossession(TypedDict, total=False):
    """Posesión esperada."""

    local: float
    visitante: float


class Aggressiveness(TypedDict, total=False):
    """Índice de agresividad."""

    local: float
    visitante: float
    total: float


class Volume(TypedDict, total=False):
    """Volumen de eventos esperado."""

    tiros_total: float
    corners_total: float
    pace_index: float


class ExpectedMetrics(TypedDict, total=False):
    """Métricas esperadas del partido."""

    xgoles: XGoals
    xfaltas: XFouls
    xposesion: XPossession
    agresividad: Aggressiveness
    volumen: Volume


class ResultOdds(TypedDict, total=False):
    """Probabilidades del resultado 1X2."""

    prob_local: float
    prob_empate: float
    prob_visitante: float
    entropia: float


class OverUnderLine(TypedDict, total=False):
    """Línea over/under."""

    linea: float
    prob_over: float
    prob_under: float


class MarketDerivatives(TypedDict, total=False):
    """Métricas derivadas del mercado."""

    market_entropy: float
    market_balance: float
    market_favorite_prob: float
    has_market_odds: bool


class MarketBlock(TypedDict, total=False):
    """Bloque completo del mercado."""

    resultado: ResultOdds
    goles_ou: OverUnderLine
    faltas_total_ou: OverUnderLine
    derivadas: MarketDerivatives


class MatchLabels(TypedDict, total=False):
    """Labels categoricos del partido."""

    intensidad_esperada: str
    riesgo_disciplinario: str
    is_derby: bool
    season_phase: float
    tipo_partido: str
    sesgo_mercado: str
    h2h_faltas_media: float
    h2h_partidos: int


class MetaBlock(TypedDict, total=False):
    """Metadatos del contrato."""

    schema_version: str
    generated_at: str
    source_snapshot: dict
    input: dict
    narrative: list[str]
    market_used_for_model: list[str]
    warnings: list[str]
    quality: dict


class FeatureContract(TypedDict, total=False):
    """Contrato completo de features para predicción.

    Resultado de _assemble_contract() — estructura anidada completa
    con todos los datos del partido, equipos, métricas y mercado.
    """

    partido: MatchInfo
    arbitro: dict  # RefereeStats + interacciones
    equipos: dict[str, TeamBlock]  # "local" y "visitante"
    metricas_esperadas: ExpectedMetrics
    mercado: MarketBlock
    contexto_partido: MatchLabels
    _meta: MetaBlock
