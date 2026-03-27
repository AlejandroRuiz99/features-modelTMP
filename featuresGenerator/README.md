# featuresGenerator — Pipeline KDD de Features para Prediccion de Faltas

Dado un partido (local, visitante, jornada, arbitro), **genera el flat dict de features** listo para el modelo predictor de faltas de La Liga. Sigue la arquitectura **KDD (Knowledge Discovery in Databases)** con fases explicitas de seleccion, transformacion, ensamblaje y evaluacion.

---

## Uso rapido

```python
from generate import generate_features, generate_features_batch

# Un partido -> flat dict listo para el modelo
feat = generate_features(
    equipo_local="Real Madrid",
    equipo_visitante="Ath Madrid",
    jornada=29,
)

# Con arbitro explicito y perfil de features personalizado
feat = generate_features(
    equipo_local="Real Madrid",
    equipo_visitante="Ath Madrid",
    jornada=29,
    fecha_partido="2026-03-22",
    features_profile="prediction",   # "prediction" | "training" | "minimal"
)

# Batch: varios partidos (carga el estado historico una sola vez)
feats = generate_features_batch([
    {"equipo_local": "Real Madrid",  "equipo_visitante": "Ath Madrid",  "jornada": 29},
    {"equipo_local": "Barcelona",    "equipo_visitante": "Sevilla",      "jornada": 29},
])

# Contrato anidado completo (para reports/debug)
from assembly.feature_assembler import build_detailed
from core.state_cache import get_state

contract = build_detailed(state=get_state(), equipo_local_input="Real Madrid", ...)
```

---

## Estructura KDD

```
featuresGenerator/
|
|-- features.yaml          <- CONFIGURACION: que features estan activas por perfil
|-- config.yaml            <- Parametros del modelo (decay, pesos, umbrales)
|-- generate.py            <- Interfaz publica: generate_features / generate_features_batch
|
|-- core/                  <- Infraestructura compartida
|   |-- config.py          #   Constantes y secretos (config.yaml + .env)
|   |-- helpers.py         #   Funciones puras: decay_weight, parse_date, safe, clip
|   |-- utils.py           #   norm_text, TEAM_ALIASES, fuzzy_name_search, etc.
|   +-- state_cache.py     #   Estado estadistico precalculado (singleton thread-safe)
|
|-- selection/             <- Fase KDD: Seleccion y adquisicion de datos (I/O)
|   |-- supabase_client.py #   Supabase CRUD
|   |-- csv_source.py      #   CSVs de football-data.co.uk
|   |-- scraper.py         #   Scraper de posesion (fbref.com)
|   |-- calendar_client.py #   Calendario multi-competicion (API externa)
|   |-- referee_resolver.py#   Resolucion de arbitros (API RFEF)
|   |-- odds_client.py     #   Cuotas de apuestas desde Supabase
|   +-- team_mapping.py    #   Normalizacion de nombres (fbref -> BD)
|
|-- transformation/        <- Fase KDD: Transformacion y calculo estadistico (sin I/O)
|   |-- iap.py             #   Rankings de agresividad ponderada (IAP)
|   |-- xfouls.py          #   Faltas esperadas (xFouls con decay + card pressure)
|   |-- xstyle.py          #   Perfil de estilo de juego por equipo
|   |-- referees.py        #   Perfiles estadisticos de arbitros
|   |-- referee_gmm.py     #   Modelo GMM bimodal de arbitros
|   |-- competitive_context.py  # Contexto competitivo + fatiga
|   |-- market.py          #   Ajuste del knowledge pack por senal de mercado
|   |-- xgoals.py          #   xGoals y probabilidades Poisson bivariate
|   |-- xposesion.py       #   Posesion esperada
|   |-- xtarjetas.py       #   Tarjetas esperadas y agresividad por volumen
|   |-- forma.py           #   Forma reciente y contexto de temporada
|   |-- match_profile.py   #   Tipo de partido, volumen de eventos, narrative
|   |-- match_labels.py    #   Labels categoricos (intensidad, riesgo disciplinario)
|   +-- knowledge_pack.py  #   Orquestador: ensambla el knowledge pack completo
|
|-- assembly/              <- Fase KDD: Ensamblaje -> flat dict para el modelo
|   |-- feature_assembler.py  # Pipeline principal (5 pasos) + flatten
|   |-- feature_registry.py   # Carga features.yaml y filtra features por perfil
|   |-- betting_odds.py       # Procesado de cuotas de apuestas (vivo e historico)
|   +-- completeness.py       # Validacion de campos criticos
|
|-- evaluation/            <- Fase KDD: Evaluacion y calibracion de modelos
|   |-- base.py            #   Metricas: MAE, RMSE, Pearson, Log-loss, Brier, BSS
|   |-- xfouls_eval.py     #   Validacion walk-forward + grid search DECAY/ALPHA
|   |-- xgoals_eval.py     #   Backtesting xGoals vs goles reales
|   |-- xposesion_eval.py  #   Backtesting posesion
|   |-- referee_eval.py    #   Valor anadido del factor arbitro
|   |-- iap_eval.py        #   Calibracion de pesos IAP
|   |-- aggressivity_eval.py  # Calibracion agresividad por volumen
|   +-- run_all.py         #   Orquestador: ejecuta todas las evaluaciones
|
+-- training_data/         <- Pipeline de generacion del dataset de entrenamiento
    |-- generator.py       #   Pipeline completo: Supabase -> features -> Parquet
    +-- __main__.py        #   CLI: python -m training_data [--output path]
```

---

## Configuracion de features (`features.yaml`)

El archivo `features.yaml` controla que grupos de features estan activos por perfil:

```yaml
active_profile: prediction   # perfil por defecto

profiles:
  prediction: { ... }   # todas las features
  training:   { ... }   # todas + cuotas historicas B365
  minimal:    { ... }   # solo features base, sin mercado ni contexto
```

**12 grupos de features**, con tipo `base` o `derived` y dependencias explicitas:

| Grupo | Tipo | Descripcion |
|-------|------|-------------|
| `identity` | base | Identificadores del partido |
| `team_season_stats` | base | Estadisticas xStyle de temporada completa |
| `rankings` | derived | Rankings IAP normalizados |
| `expected_goals` | derived | xGoals + posesion (Poisson) |
| `expected_fouls` | derived | xFouls + factores ICC |
| `aggressiveness` | derived | Agresividad por volumen |
| `forma` | derived | Forma reciente y momentum |
| `context` | derived | Contexto competitivo, fatiga, calendario |
| `referee_profile` | derived | Perfil GMM bimodal del arbitro |
| `referee_interaction` | derived | Delta historico arbitro-equipo |
| `market` | derived | Senales de cuotas (graceful_degradation) |
| `match_profile` | derived | Tipo de partido, H2H, pace index |

> **ADVERTENCIA**: No desactives grupos `base` si hay grupos `derived` que dependen de ellos. El sistema emitira una advertencia al arrancar.

---

## Parametros (`config.yaml`)

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `decay_lambda` | `0.003` | Velocidad de decay temporal en IAP y xStyle |
| `pesos.faltas` | `1.0` | Peso de faltas en el IAP |
| `pesos.amarillas` | `0.35` | Peso de amarillas en el IAP |
| `pesos.rojas` | `0.0` | Peso de rojas en el IAP (volumen, no severidad) |
| `seasons` | `[2023, 2024, 2025]` | Temporadas a incluir |
| `alpha_card_pressure` | `0.50` | Efecto presion de tarjetas en xFouls |
| `forma_ventana` | `5` | N de partidos para forma reciente |
| `home_goals_factor` | `1.18` | Factor ventaja local en xGoals (calibrado) |
| `jornadas_laliga` | `38` | Total de jornadas en la temporada |
| `market_alignment_threshold` | `0.60` | Umbral de alineacion modelo-mercado |

---

## Ingesta de datos

```python
from selection import fetch_all, ingest_stats, ingest_possession, ingest_all

ingest_stats()       # football-data.co.uk -> Supabase
ingest_possession()  # Scrape fbref.com -> Supabase
ingest_all()         # Ambos en secuencia
```

---

## Instalacion

```bash
pip install -r requirements.txt
cp .env.example .env   # anade SUPABASE_URL y SUPABASE_KEY
```
