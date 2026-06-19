# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research question

Does **dynamic judge workload at case opening** predict *procedural complexity* (`ed`) beyond what basic case-filing attributes already explain?

- **Target `ed`** (event density) = `n_activity_types / n_events`, range 0–1. Higher = more varied procedure. It is a continuous regression target.
- The analysis is run **separately for two case families**: `cv` (civil) and `cr` (criminal).

## The 4-model design (central abstraction)

Every model script (Ridge, XGBoost) trains and compares the same four nested feature sets, defined once in [src/preprocessing.py](src/preprocessing.py#L80) (`get_feature_sets`):

| Model | Features | Isolates |
| ----- | -------- | -------- |
| 1 | filing attributes only | baseline |
| 2 | + judge **workload** (no identity) | workload signal without knowing the judge |
| 3 | + `District_Judge` identity (no workload) | who the judge is |
| 4 | + identity **and** workload | full model |

Models 2 vs 4 isolate the workload contribution with and without judge identity. Do not change one script's feature grouping without the other — both consume `get_feature_sets` so they stay aligned.

The three workload features (`WORKLOAD_COLS` in preprocessing) are `open_cases_at_filing`, `aged_open_cases_at_filing`, `clearance_rate_last_180_days`, computed by [src/judge_workload.py](src/judge_workload.py). The first two are `log1p`-transformed in `load_dataset`; clearance is a bounded ratio left raw.

## Critical conventions

- **Always use `.venv/bin/python3`.** System Python lacks xgboost, shap, torch.
- **Temporal split, never random.** `temporal_split` in preprocessing trains on the oldest 80% of `case_open_date`, tests on the newest 20%. Tuning uses `TimeSeriesSplit`. This is deliberate — workload features and the target depend on time, so a random split leaks the future.
- **Workload is computed on ALL cases, then subset.** `load_dataset` calls `add_judge_workload` on the full frame before `prepare` filters to a `case_type`, so peer/clearance counts are accurate across the whole docket.
- **Leakage columns are excluded as features** (`_NON_FEATURE` set): `n_events`, `n_activity_types` (target building blocks), `ucid`, `case_open_date`, `case_close_date`, `year`, `case_type`.
- **`District_Judge` is a real judge id string.** XGBoost casts id columns to pandas `category` (`enable_categorical=True`); Ridge one-hot encodes them. There is no integer label-encoding step in the current pipeline.
- **Output files carry the target suffix** `_{TARGET}` (e.g. `docs/xgb_results_ed.json`, `docs/xgb_best_params_ed.json`, `docs/ridge_results_ed.json`). Script docstrings sometimes show the un-suffixed name.

## Data pipeline

`Event Log.csv` (~2.7 GB, project root, git-ignored) → built into `data/by_case.parquet` by [notebooks/00_eda.ipynb](notebooks/00_eda.ipynb) (chunked load, keeps closed cases, drops multi-judge rows, aggregates to case level, computes `ed`). All model code reads `data/by_case.parquet`; the parquet and the raw CSV are git-ignored.

## Common commands

```bash
cd "/Users/mariamantsurova/Desktop/University/3rd Year/federal-court-complexity"

# Tune XGBoost first (writes docs/xgb_best_params_ed.json, reused by run_xgb)
.venv/bin/python3 scripts/tune_xgb.py --n-iter 20

# Train + compare the 4 models (both case types, or one)
.venv/bin/python3 scripts/run_xgb.py                       # both cv and cr
.venv/bin/python3 scripts/run_xgb.py --case-type cv
.venv/bin/python3 scripts/run_ridge_regression.py          # alpha tuned via RidgeCV
.venv/bin/python3 scripts/run_ridge_regression.py --case-type cr --alpha 10

# Cross-model and poster figures (read the docs/*_ed.json result files)
.venv/bin/python3 scripts/compare_ridge_xgb.py

# Smoke-test the parquet
.venv/bin/python3 -c "import pandas as pd; d=pd.read_parquet('data/by_case.parquet'); print(d.shape); print(d['ed'].describe())"
```

Scripts write metrics JSON to `docs/` and figures (PNG, `matplotlib Agg` backend) to `reports/figures/`. `add_judge_workload` takes ~30–60s on the full dataset, so every `run_*` invocation has that fixed startup cost.

## Layout

- `src/` — reusable logic: `preprocessing.py` (split, feature sets, the 4-model contract), `judge_workload.py` (the workload features).
- `scripts/` — runnable entry points; each imports from `src/` and writes to `docs/` + `reports/figures/`.
- `docs/` — model results and tuned hyperparameters (JSON, committed).
- `notebooks/00_eda.ipynb` — the only notebook; owns the raw-log → parquet pipeline.

## Note on the project skill

The `run-federal-court-complexity` skill predates this 4-model refactor. Where it disagrees with the code (it describes a 2-model A/B design, `judge_open_at_filing`-style column names, and label-encoded `District_Judge_idx`), **trust the code and this file.** The current truth is the 4-model design above.
