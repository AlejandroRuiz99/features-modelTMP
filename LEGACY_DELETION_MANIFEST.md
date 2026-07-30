# Legacy Deletion Manifest — `hwfp-legacy-absorption`

Generated at the end of Batch 6 (PR6, final batch of the `hwfp-legacy-absorption`
change). Every model, feature, and training-pipeline code path that `HWFP/`
depends on has been absorbed into `HWFP/models/`, `HWFP/features/`, and
`HWFP/training/` across PR1–PR5. This document lists every legacy path that
is safe to delete, with the scan evidence proving `HWFP/` has zero incoming
references to it.

**Deletion is executed by the user, not by this change.** This is an advisory
document only — no files listed below were deleted as part of this batch.

## Scan Methodology

For each candidate path, the following searches were run against `HWFP/`
(all commands run from the repo root, matches limited to `*.py` files):

```bash
grep -rIn "<legacy-import-name>" HWFP --include="*.py"
```

A candidate is included below only if every hit found is a comment,
docstring, or string-literal path check — never a live `import`/`from ...
import` statement. Any hit that IS a live import is called out explicitly
under "Still-Referenced Paths" and excluded from the deletion list.

This is independent of (and stricter for `HWFP/` than) the AST-based
`test_no_legacy_package_imports_under_hwfp` architecture-boundary test added
this batch, which enforces the same zero-import invariant as a permanent
regression guard for `src.*`, `prediction_models*`, `features_generator*`,
and bare `assembly`/`transformation`.

## Deletion Candidates

### 1. `prediction_models/` (entire tree)

| Path | Status | Evidence |
|---|---|---|
| `prediction_models/src/models/*.py` | **Empty** — 7 model files absorbed into `HWFP/models/` in PR1 (Batch 1, commit `dedd561`) | Only `__pycache__/*.pyc` remain on disk; zero `.py` source files. `grep -rn "src\.models" HWFP` returns zero live imports (docstring/comment mentions only, e.g. `HWFP/tests/unit/test_model_checkpoint_parity.py`'s historical-context comment). |
| `prediction_models/src/utils/*.py` | **Empty** — `ev.py`/`distributions.py` absorbed into `HWFP/models/utils/` in PR1 | Same as above — only `.pyc` remain. |
| `prediction_models/config/model_config.yaml` | **Moved** — now `HWFP/models/config/model_config.yaml` (PR1, task 1.3) | `prediction_models/config/` is empty on disk. |
| `prediction_models/checkpoints/ensemble/*` | **Moved** — production checkpoint now lives at `HWFP/models/checkpoints/ensemble/` (PR1, task 1.4), sha256-verified byte-identical (`HWFP/tests/unit/test_model_checkpoint_parity.py`, 18/18 files) | Original location no longer exists under `prediction_models/checkpoints/` (only `ensemble_backup_pre_repairs/` and `eval_season/` remain — see below). |
| `prediction_models/checkpoints/ensemble_backup_pre_repairs/*` | **Never absorbed** — a pre-repair backup snapshot, explicitly excluded from PR1's move by design (task 1.4: "only production `ensemble/`, not `ensemble_backup_pre_repairs/`") | Zero references anywhere in `HWFP/`. Historical/rollback artifact only — user's call whether to keep as an offline backup or delete. |
| `prediction_models/checkpoints/eval_season/season_split_2024-25/*` | **Never absorbed** — a season-holdout evaluation snapshot, explicitly excluded from PR1's move | Zero references anywhere in `HWFP/`. |
| `prediction_models/data/` | **Empty** — `training.parquet` moved (filesystem move, gitignored — never tracked in git) to `HWFP/training/data/training.parquet` in PR5 (Batch 5) | Empty directory on disk. `HWFP/tests/unit/test_training_composition_container.py` has a `_source_parquet()` fallback that checks this path *first for existence* before falling back — this is dead code today because `HWFP/training/data/training.parquet` (the moved file) always exists, so the fallback branch never executes. Not a live dependency. |

**Verdict**: 100% of `prediction_models/` is a deletion candidate. Nothing under it is imported by `HWFP/`.

### 2. `features_generator/` (entire tree)

| Path | Status | Evidence |
|---|---|---|
| `features_generator/{assembly,transformation,core}/` | **Moved** — absorbed into `HWFP/features/{assembly,transformation,core}/` in PR2 (Batch 2, commit `742406e`), including `config.yaml`/`features.yaml` transitive dependencies | These subdirectories no longer exist under `features_generator/` at all (confirmed via directory listing — only `evaluation/`, `selection/`, `training_data/`, and root files remain). |
| `features_generator/selection/*.py` | **Deliberately NOT absorbed** (design D2: "drop `selection/`") — `selection.odds_client`'s real-fetch capability was replaced this batch (PR6) with an injectable DI hook (`HWFP.features.assembly.betting_odds.set_market_data_source`); no code path in `HWFP/` imports `selection` anymore as of this batch | Before PR6, `HWFP/features/assembly/betting_odds.py` had a call-time `from selection.odds_client import get_match_odds_rows` — this was the **one remaining external-legacy dependency** flagged since Batch 2/4/5. **Resolved in PR6**: the import was removed entirely and replaced with `set_market_data_source(fn)` DI. `grep -rn "selection\." HWFP --include="*.py"` now returns zero hits. |
| `features_generator/evaluation/*.py` | **Never absorbed** — explicitly out of scope per design D2 (evaluation/reporting tooling, not a runtime dependency of any absorbed pipeline) | Zero references in `HWFP/`. |
| `features_generator/training_data/*.py` | **Never absorbed** — explicitly out of scope per design D2 (legacy training-data generation superseded by `HWFP/training/adapters/parquet_training_data_source.py`, PR5) | Zero references in `HWFP/`. |
| `features_generator/generate.py`, `README.md`, `requirements.txt`, `.env*`, `.gitignore` | **Never absorbed** — legacy CLI entrypoint and packaging metadata for the standalone `features_generator` tool | Zero references in `HWFP/`. |

**Verdict**: 100% of `features_generator/` is a deletion candidate, including `selection/` (its one live consumer was eliminated in this batch).

### 3. Other legacy top-level directories

| Path | Status | Evidence |
|---|---|---|
| `overlay/` | **Never absorbed** — narrative/objective overlay tooling for the legacy pipeline, no equivalent exists in `HWFP/` (out of current scope) | `grep -rn "^from overlay\|^import overlay" HWFP --include="*.py"` → zero hits. |
| `staking/` | **Never absorbed** — legacy staking-simulation tooling; `HWFP/serving/adapters/kelly_staking_calculator.py` is an independent, already-absorbed implementation, not a consumer of this package | `grep -rn "^from staking\|^import staking" HWFP --include="*.py"` → zero hits. |
| `parsers/` | **Never absorbed** — legacy CLI/report parsing helpers | `grep -rn "^from parsers\|^import parsers" HWFP --include="*.py"` → zero hits. |
| `audit/` | **Never absorbed** — legacy audit-trail tooling | `grep -rn "^from audit\|^import audit" HWFP --include="*.py"` → zero hits. |
| `freshness/` | **Never absorbed** — legacy data-freshness checks | `grep -rn "^from freshness\|^import freshness" HWFP --include="*.py"` → zero hits. |
| `scripts/*.py` (`analyze_backtest.py`, `backtest_odds.py`, `overlay_backfill.py`, `rebuild_referee_profiles.py`, `run_data_miner.py`, `staking_sim.py`, `train.py`, `update_stats.py`) | **`train.py` superseded, not imported** — its pure functions (`temporal_split`, `_team_averages`, `_grid_search`, isotonic calibration, holdout eval) were **ported** (rewritten, not imported) into `HWFP/training/adapters/pytorch_model_trainer.py` in PR5. Design D3 deviation #3 explicitly rejected importing `scripts/train.py` from anything reachable via `HWFP/`'s import graph, because it mutates `sys.path` at module level (would violate REQ-14 in spirit). The rest of `scripts/*.py` was never in scope for absorption. | `grep -rn "^from scripts\|^import scripts" HWFP --include="*.py"` → zero hits. The **one exception**: `HWFP/tests/unit/test_pytorch_model_trainer.py`'s equivalence test (`test_equivalence_against_scripts_train_within_tolerance`, PR5 task 5.7) deliberately test-scope-imports `scripts.train` as a reference oracle — see "Still-Referenced Paths" below. |
| `run_prediction.py` (repo root) | **Never absorbed** — legacy CLI entrypoint superseded by `HWFP/cli/bot_main.py` | `grep -rn "run_prediction" HWFP --include="*.py"` → zero hits. |
| `odds_definitivo_25_26.csv` (repo root) | **Never absorbed** — only consumed by `scripts/backtest_odds.py` (itself a deletion candidate above) | Zero references in `HWFP/`. |

### 4. Root `tests/` — files broken by the migration

Running `pytest tests/ --collect-only` produces **24 collection errors** (421
other tests in the same tree still collect fine — they don't touch the
moved modules). The 24 broken files import legacy dotted paths that moved
during this change and no longer exist at their old location:

| Broken import | Files affected | Root cause |
|---|---|---|
| `src.models.*` (e.g. `src.models.anfis`) | `tests/unit/test_anfis_variance.py`, `tests/unit/test_calibration_load.py`, `tests/unit/test_ensemble_referee.py`, `tests/unit/test_naive_bayes.py`, `tests/unit/test_referee_gmm.py`, `tests/unit/test_referee_gmm_shrinkage.py` | Moved to `HWFP.models.*` in PR1. |
| `src.utils.*` (e.g. `src.utils.distributions`) | `tests/unit/test_distributions.py`, `tests/unit/test_distributions_smooth.py`, `tests/unit/test_negbin_alpha.py`, `tests/overlay/test_applier.py`, `tests/overlay/test_ev_kelly_scale.py`, `tests/overlay/test_run_prediction_overlay.py`, `tests/overlay/test_tilt.py`, `tests/overlay/test_tilt_floor.py`, `tests/integration/overlay/test_objective_integration.py` | Moved to `HWFP.models.utils.*` in PR1. |
| `core.utils` / `assembly.*` / `transformation.*` | `tests/unit/test_utils.py`, `tests/unit/test_feature_assembler_referee.py`, `tests/unit/test_helpers.py` | Moved to `HWFP.features.core.utils` / `HWFP.features.assembly.*` / `HWFP.features.transformation.*` in PR2. |
| `predecir_jornada` (unrelated — a Claude-skill script path, not part of this migration) | `tests/unit/test_predecir_calendario_parser.py`, `tests/unit/test_predecir_ev.py`, `tests/unit/test_predecir_filter.py`, `tests/unit/test_predecir_freshness.py`, `tests/unit/test_predecir_interactive.py`, `tests/unit/test_predecir_markdown.py` | Pre-existing gap unrelated to this change — `tests/conftest.py` adds a `~/.claude/skills/predecir-jornada/scripts` path that does not exist in this environment. Listed here for completeness since these files are also un-collectable, not because this change caused it. |

**Verdict**: the first three groups (18 files) are deletion candidates —
their production code was migrated to `HWFP/` and no longer exists at the
imported path. The `predecir_jornada` group (6 files) is a pre-existing,
unrelated environment gap and is **not** a consequence of this migration;
flagged for user awareness but not a migration-deletion candidate.

## Still-Referenced Paths (excluded from deletion)

Per spec scenario "Still-referenced path": a candidate is excluded if
`HWFP/` has at least one incoming edge to it. Exactly one file matches this
during the scan, and it is a **deliberate, test-scoped exception**, not a
production dependency:

| Path | Referenced by | Why it's excluded (partially) |
|---|---|---|
| `scripts/train.py` | `HWFP/tests/unit/test_pytorch_model_trainer.py::test_equivalence_against_scripts_train_within_tolerance` | This single test imports `scripts.train` directly and by design (D3 deviation #3) — it is the reference oracle proving `PyTorchModelTrainer.fit()`'s `nll` matches the original implementation within tolerance (`|Δnll| ≤ 0.01`, verified in PR5). **This is not a production import** — `HWFP`'s runtime code (adapters, composition, CLI) never imports `scripts.train`. If `scripts/train.py` is deleted, this one equivalence test must be deleted or rewritten to use a pinned reference value instead of a live comparison — **flagged for the user's deletion decision**, not blocking the rest of the manifest. |

No other legacy path has any incoming edge from `HWFP/` as of this batch.

## Known Gaps Outside This Manifest's Scope

Recorded here for visibility, not resolved by this batch (deletion-manifest
generation and import-boundary hardening only):

1. **Supabase environment wiring for `main()`**: `HWFP/cli/bot_main.py`'s
   `main()` never reads `SUPABASE_URL`/`SUPABASE_KEY` from the environment,
   so `SupabaseStateAdapter()`/`SupabasePerformanceTracker()` cannot be
   constructed for a real production run today (flagged since Batch 4).
2. **No real market-odds data source wired**: PR6 made market-fetch
   injectable and safe-by-default (`skip_market_fetch=True`), but no real
   Supabase-backed implementation of `HWFP.features.assembly.betting_odds
   .set_market_data_source(...)` exists yet — `main()` still needs one
   wired at composition time before `skip_market_fetch=False` is usable in
   production.
3. **`HWFP/cli/bot_main.py::_unavailable_partidos_source`**: placeholder
   that raises `RuntimeError` — no real production "raw partidos" fetcher
   is wired into `state_cache.set_data_source(...)` yet (flagged since
   Batch 4).

## Summary

| Category | Verdict |
|---|---|
| `prediction_models/` (entire tree) | Delete |
| `features_generator/` (entire tree, incl. `selection/`) | Delete |
| `overlay/`, `staking/`, `parsers/`, `audit/`, `freshness/` (repo root) | Delete |
| `scripts/*.py` (all 8 files) | Delete — see exception below for the one test dependency |
| `run_prediction.py`, `odds_definitivo_25_26.csv` (repo root) | Delete |
| `tests/unit/test_{anfis_variance,calibration_load,distributions,distributions_smooth,ensemble_referee,feature_assembler_referee,helpers,naive_bayes,negbin_alpha,referee_gmm,referee_gmm_shrinkage,utils}.py`, `tests/overlay/test_*.py` (8 files), `tests/integration/overlay/test_objective_integration.py` | Delete (18 files total — broken imports to moved code) |
| `scripts/train.py` | Delete only after removing/rewriting `test_equivalence_against_scripts_train_within_tolerance` |
| `tests/unit/test_predecir_*.py` (6 files) | Not this migration's concern — pre-existing unrelated gap |

Deletion is executed by the user. This manifest does not modify or remove
any file.
