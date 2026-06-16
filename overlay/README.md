# Overlay — Context-Aware Foul Prediction System

The `overlay/` package applies a narrative-driven overlay to the foultsPredictor ensemble.
It introduces three injection points in `run_prediction.py`:

- **P1 (pre-feature)**: Objective injection — patches `state['objectives']` before feature generation.
- **P3 (post-model)**: PMF tilt — shifts the output probability distribution by `delta_fouls`.
- **P4 (post-EV)**: Kelly scale — applies a confidence multiplier to the bet sizing.

**Key design principle**: Narratives now REQUIRE objectives for both teams. The `laliga_objectives` Supabase fallback was removed in D17 — objectives come exclusively from the narrative YAML.

> ### D17 Migration Note (April 2026)
>
> - The field `objective_override` was renamed to `objectives` and is now **REQUIRED** (both `home` and `away` keys mandatory).
> - The Supabase function `fetch_laliga_objectives()` has been **removed**. Objectives now come exclusively from narrative YAML files via the overlay P1 injection.
> - The function `apply_objective_override()` has been renamed to `inject_objectives_into_state()`. The old name is kept as a deprecated alias and will be removed in a future batch.
> - All 9 backfill narratives in `overlay/backfill/narratives/` have been migrated to the new schema.
> - Backtest scripts that depended on `fetch_laliga_objectives()` are an **out-of-scope follow-up** (see `predecir-jornada-v2-backtest-migration` in the roadmap).

---

## Quick Start

### Single match with narrative
```bash
python run_prediction.py \
  --local "Espanyol" --visitante "Levante" \
  --jornada 32 --fecha 2026-04-27 \
  --narrative overlay/backfill/narratives/espanyol_vs_levante_2026-04-27.yaml \
  --overlay-log-dir overlay/logs
```

### Batch with narratives directory
```bash
python run_prediction.py \
  --batch-file partidos_hoy.json \
  --narratives overlay/backfill/narratives \
  --overlay-log-dir overlay/logs \
  --output-json resultados_hoy.json
```

### Validate a narrative YAML (no prediction)
```bash
python run_prediction.py --validate-narrative path/to/narrative.yaml
```

### Backfill calibration report
```bash
python scripts/overlay_backfill.py \
  --narratives overlay/backfill/narratives \
  --actuals overlay/backfill/actuals.json \
  --output overlay/backfill/initial_calibration_report.md
```

---

## Narrative YAML Schema

Each narrative YAML file annotates a single match. All fields except `match` and
`confidence_level` are optional. Unknown fields raise a `ValueError` at parse time.

> **Note**: Files in `overlay/backfill/narratives/` are **DRAFT** — they were
> generated from historical records and need user review before use in production.

### Full example with all fields annotated

```yaml
# DRAFT — review before production use   ← always start with this comment for drafts

match:                                    # Required — identifies the match
  home: "Espanyol"                        # Required string — home team name
  away: "Levante"                         # Required string — away team name
  date: "2026-04-27"                      # Required ISO date YYYY-MM-DD
  competition: "La Liga"                  # Optional, default: "La Liga"
  jornada: 32                             # Optional int — matchday number

objectives:                               # REQUIRED (D17) — both home and away mandatory
  home:                                   # REQUIRED: "home" key
    label: salvacion                      # Required: one of VALID_LABELS (see below)
    urgency_base: 0.65                    # Optional float [0.0, 1.0] — explicit urgency
  away:                                   # REQUIRED: "away" key
    label: descenso
    urgency_base: 0.80

stakes:                                   # Optional — perceived match stakes per team
  home: 4                                 # int [0, 5] — 0 = meaningless, 5 = maximum
  away: 5
  notes: "Relegation battle"              # Optional free text

rotations:                                # Optional — number of players rested
  home: 0                                 # int [0, 5] — 0 = full strength, 5 = B-team
  away: 1

intensity_override: 4                     # Optional int [0, 5] — expected match intensity
physicality_bias: 1                       # Optional int [-2, +2] — physical contact bias
referee_factor: 0                         # Optional int [-2, +2] — ref strictness adj.

special_flags:                            # Optional list — closed enum (see below)
  - stakes_both_relegation
  - late_season
  - physical_clash

confidence_level: 4                       # Required int [1, 5] — your confidence in the narrative

notes: "Free text summary for humans"     # Optional
```

### Valid labels for `objectives.*.label`

| Label | Meaning | Default urgency |
|-------|---------|-----------------|
| `titulo` | Title race | 0.85 |
| `ucl` | Champions League qualification | 0.75 |
| `europa` | Europa / Conference League | 0.60 |
| `mid` | Mid-table, no objective pressure | 0.25 |
| `salvacion` | Survival (informal relegation pressure) | 0.80 |
| `descenso` | Relegation zone | 0.80 |

If `urgency_base` is omitted, the system uses the default from the table above.

### Valid `special_flags`

| Flag | When to use |
|------|-------------|
| `stakes_both_relegation` | Both teams in relegation fight |
| `stakes_one_relegation` | One team in relegation, other has stakes too |
| `derbi` | Local derby or intense rivalry |
| `copa_recent_extra_time` | A key team played Copa del Rey extra time ≤5 days ago |
| `european_midweek` | UEFA match in the midweek prior |
| `coach_debut` | New coach's first match |
| `coach_pressure` | Coach under sacking pressure |
| `last_round_drama` | Final matchday with multiple scenarios |
| `weather_extreme` | Extreme weather expected |
| `b_team_expected` | B-team or significant rotation expected |
| `key_injuries_home` | Multiple key home injuries |
| `key_injuries_away` | Multiple key away injuries |
| `morbo` | High tension / political context / enmity |
| `dead_rubber` | Meaningless match for both teams |
| `late_season` | Final weeks of the season (typical ≥J30) |
| `early_season` | Opening weeks (typical ≤J5) |
| `physical_clash` | Known physical styles that typically collide |
| `strict_ref_announced` | Verified strict referee designation |
| `permissive_ref_announced` | Verified permissive referee designation |

---

## Rule Catalog Summary (15 rules)

Rules live in `overlay/rules.yaml`. Each rule has:
- `id`: snake_case unique identifier
- `direction`: `up` | `down` | `volatility`
- `enabled`: `true` | `false` (can be toggled without code changes)
- `when`: DSL condition block (all / any / not + leaf operators)
- `effect`: `delta_fouls` (±), `variance_scale` (≥1.0), `kelly_scale` (≤1.0)

### UP rules (more fouls expected)

| ID | Description | Δfouls | var | kelly |
|----|-------------|--------|-----|-------|
| `both_relegation_up` | Both teams fighting relegation | +0.8 | 1.15 | 0.85 |
| `one_relegation_high_stakes_up` | One team relegated + other high stakes | +0.5 | 1.10 | 0.90 |
| `derbi_intensity_up` | Derby match | +0.6 | 1.10 | 0.90 |
| `physical_clash_up` | Physical clash flag + physicality_bias ≥ 1 | +0.5 | 1.05 | 0.95 |
| `coach_pressure_up` | Coach under pressure + intensity ≥ 3 | +0.4 | 1.10 | 0.90 |
| `last_round_drama_up` | Final matchday, both stakes ≥ 4 | +0.7 | 1.20 | 0.80 |
| `strict_ref_physical_up` | Strict ref + physical match | +0.4 | 1.05 | 0.95 |
| `high_intensity_override_up` | intensity ≥ 4 AND physicality_bias ≥ 1 | +0.6 | 1.10 | 0.90 |

### DOWN rules (fewer fouls expected)

| ID | Description | Δfouls | var | kelly |
|----|-------------|--------|-----|-------|
| `dead_rubber_down` | Meaningless match, low stakes | −0.8 | 1.10 | 0.85 |
| `heavy_rotations_both_down` | Both teams rotating ≥ 3 | −0.7 | 1.10 | 0.85 |
| `one_team_rotation_down` | One team rotating ≥ 4 | −0.4 | 1.05 | 0.90 |
| `permissive_ref_technical_down` | Permissive ref + low physicality | −0.5 | 1.05 | 0.95 |
| `b_team_expected_down` | B-team or major rotations | −0.6 | 1.10 | 0.80 |
| `european_fatigue_down` | European midweek + minimal rotations | −0.4 | 1.10 | 0.90 |
| `copa_extra_time_fatigue_down` | Recent Copa extra time | −0.3 | 1.15 | 0.85 |

### Effect bounds

| Effect | Per-rule range | Aggregate cap |
|--------|---------------|---------------|
| `delta_fouls` | [−1.0, +1.0] | ±2.0 |
| `variance_scale` | [1.0, 1.5] | [0.67, 1.5] |
| `kelly_scale` | [0.25, 1.0] | [0.25, 1.0] |

### Directional gate (REQ-3.4)

The aggregate `delta_fouls` is only applied when:
1. `confidence_level ≥ 3`
2. At least 2 fired rules share the same sign as the aggregate delta

Otherwise `delta_fouls = 0` but `variance_scale` and `kelly_scale` still apply.

---

## How to Add / Modify Rules

1. Open `overlay/rules.yaml`.
2. Add a new entry under `rules:` following the schema above.
3. Test: `pytest tests/overlay/test_catalog_loader.py tests/overlay/test_rules_dsl.py -v`
4. Constraint: `|delta_fouls| ≤ 1.0`, `variance_scale ∈ [1.0, 1.5]`, `kelly_scale ∈ [0.25, 1.0]`

### To disable a rule without deleting it
```yaml
- id: copa_extra_time_fatigue_down
  enabled: false          # ← set to false
  ...
```

---

## Overlay Log Files

Every match with a narrative produces a JSON log at `overlay/logs/` (configurable via `--overlay-log-dir`).

**Naming**: `{date}_{home}_vs_{away}.json` (collision-safe: adds `_HHMMSS` suffix)

**Schema**:
```json
{
  "timestamp": "2026-04-27T12:00:00+00:00",
  "match": { "home": "Espanyol", "away": "Levante", "date": "2026-04-27" },
  "narrative_raw": "# auto-generated\nconfidence_level: 4\n",
  "parsed_flags": {
    "confidence_level": 4,
    "special_flags": ["stakes_both_relegation", "physical_clash"],
    "objectives": {
      "home": {"label": "salvacion", "urgency_base": 0.65}
    },
    "stakes": {"home": 4, "away": 5},
    "rotations": null,
    "intensity_override": 4,
    "physicality_bias": 1,
    "referee_factor": 0
  },
  "pre_overlay": { "expected_fouls": 25.9, "pmf_summary": {...} },
  "rules_fired": [
    {"id": "both_relegation_up", "direction": "up", "magnitude_applied": 0.8, "suppressed_by_floor": false}
  ],
  "post_overlay": { "expected_fouls": 27.9, "pmf_summary": {...} },
  "kelly_raw_vs_scaled": { "kelly_raw": 1.0, "kelly_scaled": 0.7225 },
  "actual_fouls": null
}
```

### Filling actual fouls after the match
```bash
python -m overlay.fill_actuals overlay/logs/2026-04-27_Espanyol_vs_Levante.json 22
# Use --force to overwrite an existing value
python -m overlay.fill_actuals overlay/logs/2026-04-27_Espanyol_vs_Levante.json 25 --force
```

---

## Backfill Workflow

The backfill script re-runs (or attempts to re-run) historical predictions with narratives
and compares pre-overlay vs post-overlay against actual fouls.

```bash
# 1. Write narrative YAMLs to overlay/backfill/narratives/ (or pass file list)
# 2. Populate overlay/backfill/actuals.json with {match_key: actual_fouls}
#    Key format: "{Home}_vs_{Away}_{YYYY-MM-DD}" (spaces → underscores)
# 3. Run:
python scripts/overlay_backfill.py \
  --narratives overlay/backfill/narratives \
  --actuals overlay/backfill/actuals.json \
  --output overlay/backfill/initial_calibration_report.md
```

The report is a markdown table with columns:
`match | pre_pred | post_pred | actual | line_pre | line_post | hit_pre | hit_post | rules_fired`

- `actual = NA` when no actuals provided (acceptable for initial drafts).
- Invalid or unreadable narratives are **skipped** and listed at the bottom of the report.

> **Note on predictions in offline backfill**: Running predictions requires a live Supabase
> connection to load historical state. If run offline, `pre_pred` and `post_pred` will show
> `NA`. Fill `actuals.json` with known results and re-run when Supabase is available.

---

## Schema Reference (for LLM-parser prompts)

Paste this block into your LLM prompt to generate valid narrative YAMLs:

```
Generate a narrative YAML for a La Liga match. Schema:

Required fields:
  match.home: str          # Team name (canonical: use full name)
  match.away: str
  match.date: str          # YYYY-MM-DD
  confidence_level: int    # 1-5 (use 3-4 for typical rich narratives)

Optional fields:
  match.competition: str   # default "La Liga"
  match.jornada: int       # matchday 1-38

  objectives:      # patch team objectives
    home|away:
      label: titulo|ucl|europa|mid|salvacion|descenso
      urgency_base: float  # 0.0-1.0, omit to use label default

  stakes:                  # perceived match importance
    home: int              # 0=none, 5=maximum
    away: int
    notes: str             # optional

  rotations:               # players rested
    home: int              # 0=full strength, 5=B-team
    away: int

  intensity_override: int  # 0-5
  physicality_bias: int    # -2 to +2
  referee_factor: int      # -2 to +2

  special_flags: list      # any of:
    stakes_both_relegation, stakes_one_relegation, derbi,
    copa_recent_extra_time, european_midweek, coach_debut,
    coach_pressure, last_round_drama, weather_extreme,
    b_team_expected, key_injuries_home, key_injuries_away,
    morbo, dead_rubber, late_season, early_season,
    physical_clash, strict_ref_announced, permissive_ref_announced

  notes: str               # free text for humans

Rules:
- confidence_level must be 1-5
- stakes must be 0-5 per team
- rotations must be 0-5 per team
- intensity_override must be 0-5
- physicality_bias and referee_factor must be -2 to +2
- objectives keys must be "home" or "away" only
- special_flags must be from the closed enum above
- Unknown fields raise ValueError (fail fast)
```

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `overlay/schema.py` | Dataclasses + validation for narrative YAML |
| `overlay/loader.py` | YAML discovery, parsing, error handling |
| `overlay/rules.py` | Rule catalog loading, DSL evaluation, aggregation |
| `overlay/rules.yaml` | Rule catalog (15 rules) |
| `overlay/objective.py` | P1: objective override patch for state['objectives'] |
| `overlay/tilt.py` | P3: PMF tilt via distributions.py utilities (FROZEN) |
| `overlay/kelly.py` | P4: Kelly scale computation |
| `overlay/applier.py` | Orchestrates P3 + P4 post-prediction pipeline |
| `overlay/log_writer.py` | JSON log writer (atomic, collision-safe) |
| `overlay/fill_actuals.py` | CLI to fill actual_fouls in log files |
| `scripts/overlay_backfill.py` | Batch backfill calibration report |

---

## Testing

```bash
# All overlay unit tests
pytest tests/overlay/ -v

# Integration tests
pytest tests/integration/overlay/ -v

# Identity regression (MUST stay green — distributions.py is FROZEN)
pytest tests/overlay/test_run_prediction_identity.py -v

# Full overlay suite
pytest tests/overlay tests/integration/overlay -v
```

Current passing count: **189+ tests**.

---

*Last updated: 2026-04-29 — ajuste-senal-contextual change.*
