---
name: run-federal-court-complexity
description: Run, build, test, or smoke-test the federal court complexity pipeline. Use when asked to run scripts, train models, run the pipeline, test preprocessing, or verify results.
---

# Federal Court Complexity Pipeline

**Research question:** Does dynamic judge workload at case opening predict *procedural complexity* (`complexity_index`) beyond what basic case filing attributes already explain?

Two models are compared for civil and criminal cases separately:
- **Model A** — filing features only (~30 case attributes known at filing time)
- **Model B** — filing features + `judge_workload_at_open` (concurrent open cases for that judge on the filing date)

**Stack:** pandas, scikit-learn, XGBoost, SHAP, scipy  
**Venv:** `.venv/` at project root — always use `.venv/bin/python3`  
**Data:** Pre-built parquet files live in `data/` (no raw CSV needed for most tasks)

---

## Prerequisites

- `.venv/` already exists and is fully installed
- `data/case_features.parquet` (168k rows) — already present, not in git
- `data/aggregations/by_case.parquet` — already present (has `complexity_index` column added by `build_aggregations.py`)
- `Event Log.csv` (~2.7 GB) — required **only** for Step 1 rebuild from scratch

---

## Scripts that exist and work

| Path | Purpose |
|---|---|
| `scripts/build_features.py` | Step 1: `Event Log.csv` → `Event Log_model.csv` |
| `src/build_case_features.py` | Step 2: `Event Log_model.csv` → `data/case_features.parquet` |
| `src/build_aggregations.py` | Step 3: `case_features.parquet` → `data/aggregations/by_case.parquet` |
| `scripts/run_baseline.py` | Judge-median baseline for `complexity_index` |
| `scripts/run_rf_shap.py` | RF: Model A vs Model B, with SHAP |
| `scripts/run_xgb_shap.py` | XGBoost: Model A vs Model B, with SHAP |
| `scripts/run_pipeline.py` | Orchestrates all steps end-to-end |
| `src/features.py` | Feature definitions, `TARGET = "complexity_index"`, `add_derived_columns()` |
| `src/preprocessing_trees.py` | Builds `(X, y)` for tree models; `include_workload=True` for Model B |
| `src/judge_workload.py` | Computes `judge_workload_at_open` from `case_open_date` + `los_days` |
| `src/suit_features.py` | Extracts suit-structure features from `nature_suits` array |

**Deleted (no longer exist):** `scripts/run_neural_net.py`, `src/neural_net_model.py`, `src/preprocessing_neural_net.py`, `src/judge_vocabulary.py`, `notebooks/05_hyperparameter_tuning.ipynb`

---

## Run the pipeline

### Full pipeline (from aggregations onward)

```bash
cd "/Users/mariamantsurova/Desktop/University/3rd Year/federal-court-complexity"

# Skip data rebuild steps (parquets already exist), run both models for cv + cr
.venv/bin/python3 scripts/run_pipeline.py --skip-clean --skip-case --skip-agg
```

### Individual model scripts

```bash
# Random Forest — civil cases
.venv/bin/python3 scripts/run_rf_shap.py --case-type cv

# Random Forest — criminal cases
.venv/bin/python3 scripts/run_rf_shap.py --case-type cr

# XGBoost — civil cases
.venv/bin/python3 scripts/run_xgb_shap.py --case-type cv

# XGBoost — criminal cases
.venv/bin/python3 scripts/run_xgb_shap.py --case-type cr
```

Each script runs **both Model A and Model B** and reports `workload_mae_improvement`.

Outputs:
- `docs/01_rf_results_{cv|cr}.json` — MAE, R², top SHAP features, model A vs B comparison
- `docs/02_xgb_results_{cv|cr}.json` — same for XGBoost
- `reports/figures/01_rf_shap_model_*.png` — SHAP bar plots
- `reports/figures/02_xgb_shap_*.png` — SHAP bar + beeswarm plots

### Baseline

```bash
.venv/bin/python3 scripts/run_baseline.py --case-type cv
.venv/bin/python3 scripts/run_baseline.py --case-type cr
```

Output: `docs/00_baseline_results_{cv|cr}.json`

---

## Smoke-test preprocessing

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.features import add_derived_columns
from src.preprocessing_trees import prepare_for_trees

df = pd.read_parquet('data/aggregations/by_case.parquet')
df = add_derived_columns(df)
cv = df[df['case_type'] == 'cv']

X_a, y = prepare_for_trees(cv, include_workload=False)
print('Model A:', X_a.shape, '  target median:', round(y.median(), 3))
"
```

Expected: `Model A: (151640, 233)  target median: ~-0.06`

---

## Preprocessing summary

### Target

`complexity_index` — mean z-score of `n_events`, `n_activity_types`, `n_motions`, `activity_entropy`.  
Computed in `src/features.py → add_derived_columns()`. Already present in `data/aggregations/by_case.parquet`.

**These metrics are NEVER input features** — they are retrospective (measured after case closes). Using them as features would be endogenous to the target.

### Feature groups (Model A)

| Group | Features |
|---|---|
| Plaintiff attributes | `plaintiffs_count`, `plaintiffs_share_ind/pro_se/pro_hac_vice`, `plaintiffs_counsels_count` |
| Defendant attributes | `Defendants_count`, `Defendants_share_ind/pro_se/pro_hac_vice`, `Defendants_counsels_count`, `Defendants_pending_counts`, `Defendants_terminated_counts` |
| Case structure | `Other_courts`, `related_case_count`, `Magistrate_Judge` (boolean) |
| Party types | `Party_Amicus/Counter_Claimant/Counter_Defendant/Court_Monitor/Intervenor/Material_Witness/Third_Party_Defendant/Third_Party_Plaintiff/Trustee` |
| Suit structure | `n_unique_suits`, `suit_entropy`, `is_multisuit`, `suit_dominance`, `has_<suit>`, `suit_freq_<suit>` (from `nature_suits`) |
| Case flags | `is_cv` (binary), `is_mdl_flag` (binary) |
| City | One-hot encoded |

### Model B adds

| Feature | Source |
|---|---|
| `judge_workload_at_open` | `src/judge_workload.py` — number of concurrent open cases for the same judge on `case_open_date` |

### Explicitly excluded from X

`n_events`, `n_activity_types`, `n_motions`, `activity_entropy`, `complexity_index` (→ these are the target),  
`sum_attribute_*` (retrospective event counts),  
`los_days`, `log_los_days` (outcome, not input),  
`ucid`, `case_open_date`, `District_Judge` (identifiers)

---

## Judge workload feature

`src/judge_workload.py → add_judge_workload(df)` adds `judge_workload_at_open`:

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.features import add_derived_columns
from src.judge_workload import add_judge_workload

df = pd.read_parquet('data/aggregations/by_case.parquet')
df = add_derived_columns(df)
df = add_judge_workload(df)
print(df['judge_workload_at_open'].describe())
"
```

Note: runs a vectorised interval-overlap computation grouped by judge. With 168k cases it takes ~30–60s.

---

## Data files

| File | Needed? | Notes |
|---|---|---|
| `data/case_features.parquet` | Yes | Input for `build_aggregations.py` |
| `data/aggregations/by_case.parquet` | Yes | Main model input (has `complexity_index`) |
| `data/aggregations/by_judge.parquet` | No | Not used by any current script |
| `Event Log.csv` | Only for Step 1 | ~2.7 GB, not in repo |
| `Event Log_model.csv` | Only for Step 2 | Built by `build_features.py` |

---

## Gotchas

- **`.venv/bin/python3` is required.** System Python lacks xgboost and shap.
- **`data/aggregations/by_case.parquet` has `complexity_index`; `data/case_features.parquet` does not.** Always call `add_derived_columns()` after loading, or use `by_case.parquet` which already has it.
- **`add_judge_workload()` must be called before `temporal_split()`** in model scripts — the workload is computed from the full dataset (needs all cases to count overlapping open cases).
- **Models are always run per case type** (`--case-type cv` or `--case-type cr`). Running pooled (no flag) is supported but not the primary analysis.
- **`sys.path.insert(0, str(ROOT))` is required** when calling `src/` modules from `-c` snippets.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | Use `.venv/bin/python3`, not system python3 |
| `KeyError: 'complexity_index'` | Call `add_derived_columns(df)` before `prepare_for_trees()` |
| `FileNotFoundError: data/aggregations/by_case.parquet` | Run `src/build_aggregations.py` first |
| `FileNotFoundError: Event Log.csv` | Raw log not in repo — needed only for Step 1 rebuild |
| `ModuleNotFoundError: No module named 'src.suit_features'` | Run from project root or add `sys.path.insert(0, str(ROOT))` |
