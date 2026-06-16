# Regression Snapshots — Overlay Identity Contract

These JSON snapshots are the regression contract for:
> "No narrative supplied → output byte-identical to pre-change behaviour"

## Files

| File | Match | Jornada |
|------|-------|---------|
| `espanyol_levante_j32.json` | Espanyol vs Levante | J32 2026-04-27 |
| `realmadrid_atletico_j29.json` | Real Madrid vs Atletico Madrid | J29 2026-03-22 |
| `villarreal_osasuna_j25.json` | Villarreal vs Osasuna | J25 2026-02-15 |

## How to regenerate

Run from the project root:

```bash
python tests/fixtures/snapshots/capture_snapshots.py
```

This script uses synthetic feature fixtures (no Supabase, no network). It loads the
ensemble from `prediction_models/checkpoints/ensemble/` and writes three JSON files.

**Commit the regenerated snapshots** after reviewing the diffs — they lock the regression
baseline for every test that asserts "no narrative → identical output".

## Regenerate when

- The ensemble model is retrained
- Feature engineering changes affect any snapshot fixture field
- A new field is added to `_prediction_to_dict()` in `run_prediction.py`

## Tests that use these snapshots

- `tests/integration/overlay/test_run_prediction_identity.py` (Phase 7, T7.4)
- `tests/integration/overlay/test_snapshot_regression.py` (Phase 9, T9.3)
