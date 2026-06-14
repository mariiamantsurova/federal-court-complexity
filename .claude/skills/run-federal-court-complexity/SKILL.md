---
name: run-federal-court-complexity
description: Run, build, test, or smoke-test the federal court complexity pipeline. Use when asked to run scripts, train models, run the pipeline, test preprocessing, or verify results.
---

# Federal Court Complexity Pipeline

**Research question:** Does dynamic judge workload at case opening predict _procedural complexity_ (`ed`) beyond what basic case filing attributes already explain?

**Target:** `ed` (event density) = `n_activity_types / n_events` — measures how diverse the procedural activity is relative to total docket size. Ranges 0–1; higher = more varied procedure.

Two models are compared:

- **Model A** — filing features only (case attributes known at opening)
- **Model B** — filing features + judge workload features (`judge_open_at_filing`, `judge_opened_30d`, `judge_closed_30d`)

**Stack:** pandas, scikit-learn, XGBoost, SHAP, PyTorch, scipy  
**Venv:** `.venv/` at project root — always use `.venv/bin/python3`  
**Raw data:** `Event Log.csv` (~2.7 GB) at project root

---

## Files that exist and work

| Path                                | Purpose                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------- |
| `notebooks/00_eda.ipynb`            | EDA + data pipeline: `Event Log.csv` → `data/by_case.parquet`           |
| `src/judge_workload.py`             | Computes `judge_open_at_filing`, `judge_opened_30d`, `judge_closed_30d` |
| `data/by_case.parquet`              | Case-level model input (built by notebook cell 30)                      |
| `data/aggregations/by_case.parquet` | Older aggregation (may be stale)                                        |
| `data/district_judge_lookup.json`   | Maps `District_Judge_idx` integers → judge ID strings                   |

**No model scripts exist yet** — model training has not been implemented. Only EDA and preprocessing are complete.

---

## Data pipeline (notebook `00_eda.ipynb`)

The notebook builds `data/by_case.parquet` from the raw event log:

1. **Chunked load** (cell 8): reads `Event Log.csv` in 250k-row chunks
   - Keeps only `case_status == "closed"` rows
   - `District_Judge` NaN → `"Unknown"`; multi-judge rows (comma-separated) are **dropped**
   - `Magistrate_Judge` → boolean (True if a real judge name is present)
2. **Case-level aggregation** (cell 9):
   - `n_events` = total docket entries per case
   - `n_activity_types` = unique activity types per case
   - `ed` = `n_activity_types / n_events` ← **target**
   - `case_open_date` = min `date_filed`, `case_close_date` = max `date_filed`
3. **Encoding & save** (cells 29–30):
   - `District_Judge` → `District_Judge_idx` (label encoding, lookup in `data/district_judge_lookup.json`)
   - `city` → one-hot columns
   - log1p on: `plaintiffs_count`, `plaintiffs_counsels_count`, `Defendants_count`, `Defendants_counsels_count`, `Defendants_pending_counts`, `related_case_count`
   - Saves `data/by_case.parquet`

### Smoke-test the saved parquet

```bash
cd "/Users/mariamantsurova/Desktop/University/3rd Year/federal-court-complexity"
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_parquet('data/by_case.parquet')
print(df.shape, df.dtypes.to_dict())
print(df['ed'].describe().round(4))
"
```

---

## Judge workload features

`src/judge_workload.py → add_judge_workload(df)` adds three features:

| Feature                | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| `judge_open_at_filing` | Cases same judge had open on focal case's filing date (excluding focal case) |
| `judge_opened_30d`     | Cases same judge opened in the 30 days before filing                         |
| `judge_closed_30d`     | Cases same judge closed in the 30 days before filing                         |

Requires columns: `District_Judge`, `case_open_date`, `case_close_date`.

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.judge_workload import add_judge_workload

df = pd.read_parquet('data/by_case.parquet')
df = add_judge_workload(df)
print(df[['judge_open_at_filing', 'judge_opened_30d', 'judge_closed_30d']].describe())
"
```

Note: vectorised interval-overlap grouped by judge — ~30–60s on 168k cases.

---

## Available models (from requirements.txt)

Since `ed` is a **continuous regression target** (0–1), all of the following are applicable:

| Model                       | Library   | Notes                                                                |
| --------------------------- | --------- | -------------------------------------------------------------------- |
| **Random Forest Regressor** | `sklearn` | Good baseline; handles mixed types; gives feature importance         |
| **XGBoost Regressor**       | `xgboost` | Typically strongest; use with SHAP for explainability                |
| **Ridge / Lasso**           | `sklearn` | Linear baseline; fast to fit                                         |
| **Neural Network**          | `torch`   | Embedding layer for `District_Judge_idx` → rich judge representation |
| **Gradient Boosting**       | `sklearn` | Alternative to XGBoost                                               |

**Not applicable here:**

- `lifelines` (survival analysis) — `ed` is not a time-to-event outcome
- `sentence-transformers` — judge names are already integer-encoded; no raw text features remain
- `statsmodels` — useful for significance testing of model A vs B difference, not for fitting

### Recommended order

1. Ridge regression — linear baseline, establish floor
2. XGBoost — primary model, use SHAP for feature attribution
3. Neural network with `District_Judge_idx` embedding — if judge identity proves important in XGBoost SHAP

---

## Feature groups

### Model A (filing attributes)

| Group          | Columns                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Plaintiff      | `plaintiffs_count` (log1p), `plaintiffs_share_ind/pro_se/pro_hac_vice`, `plaintiffs_counsels_count` (log1p)                                      |
| Defendant      | `Defendants_count` (log1p), `Defendants_share_ind/pro_se/pro_hac_vice`, `Defendants_counsels_count` (log1p), `Defendants_pending_counts` (log1p) |
| Case structure | `related_case_count` (log1p), `Other_courts`, `Magistrate_Judge` (bool), `is_mdl`                                                                |
| Case type      | `case_type` (cv/cr)                                                                                                                              |
| Judge identity | `District_Judge_idx` (label-encoded integer)                                                                                                     |
| City           | `city_*` (one-hot)                                                                                                                               |

### Model B adds

| Feature                | Source                  |
| ---------------------- | ----------------------- |
| `judge_open_at_filing` | `src/judge_workload.py` |
| `judge_opened_30d`     | `src/judge_workload.py` |
| `judge_closed_30d`     | `src/judge_workload.py` |

### Excluded (retrospective / identifiers)

`n_events`, `n_activity_types` (→ used to compute `ed`, the target),  
`ucid`, `case_open_date`, `case_close_date` (identifiers / time leakage)

---

## Data files

| File                              | Status              | Notes                                  |
| --------------------------------- | ------------------- | -------------------------------------- |
| `Event Log.csv`                   | Present, not in git | ~2.7 GB raw event log                  |
| `data/by_case.parquet`            | Present             | Main model input — built by notebook   |
| `data/district_judge_lookup.json` | Present             | `District_Judge_idx` → judge ID string |

---

## Gotchas

- **`.venv/bin/python3` is required.** System Python lacks xgboost and shap.
- **Multi-judge rows are dropped** in the chunked load (rows where `District_Judge` contains `", "`). This is intentional.
- **`District_Judge_idx` is label-encoded**, not ordinal — tree models and neural nets can use it; linear models should not use it as a numeric feature.
- **`add_judge_workload()` needs `District_Judge` (string), not `District_Judge_idx`** — call it before encoding, or on the unencoded `data/by_case.parquet` which retains the original string column... check if it was dropped in the save step.
- **`sys.path.insert(0, str(ROOT))` is required** when calling `src/` modules from `-c` snippets.

---

## Troubleshooting

| Error                                            | Fix                                                                                                      |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'xgboost'` | Use `.venv/bin/python3`, not system python3                                                              |
| `FileNotFoundError: Event Log.csv`               | Raw log must be at project root (~2.7 GB)                                                                |
| `KeyError: 'District_Judge'`                     | Column was dropped after encoding — use `District_Judge_idx` or reload from parquet before workload step |
| Kernel frozen at `cases.isnull().sum()`          | Restart kernel and re-run from cell 8 (chunked load) — `df` or `cases` may have been lost                |
