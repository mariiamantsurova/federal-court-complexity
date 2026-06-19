<img src="tel-aviv-university-tau4054.jpg" alt="Tel Aviv University" width="220">

# Federal Court Complexity


**Does dynamic judge workload at case opening predict the *procedural complexity* of a case, beyond what basic case-filing attributes already explain?**

A study of ~168,000 closed U.S. federal district court cases, modelling procedural complexity from attributes known at filing and testing whether a judge's live caseload adds any predictive signal.

> Research project conducted at **Tel Aviv University**, supervised by **Dr. Shany Azaria**.

---

## Research question

Procedural complexity is measured by **event density** (`ed`):

```
ed = n_activity_types / n_events
```

the number of *distinct* docket activity types relative to the *total* number of docket entries (range 0–1). A higher value means the case's procedural activity is more varied; a lower value means the docket is large but repetitive.

The central hypothesis is that a judge under heavier or "aging" caseload pressure handles cases differently, producing measurably different procedural complexity. We test this by isolating the contribution of **dynamic judge workload** — caseload computed at each case's filing date — from both static case attributes and the identity of the judge.

## Experimental design: four nested models

Every model is trained as a set of four nested feature levels, separately for **civil (`cv`)** and **criminal (`cr`)** cases:

| Model | Features | Question it answers |
| ----- | -------- | ------------------- |
| **1** | filing attributes only | baseline — what's knowable at filing |
| **2** | + judge workload (no identity) | does workload help *without* knowing the judge? |
| **3** | + judge identity (no workload) | does *who the judge is* help? |
| **4** | + identity **and** workload | full model |

Comparing **2 vs 1** and **4 vs 3** isolates the judge-workload signal with and without judge identity held constant.

### Judge workload features

Computed point-in-time at each case's filing date, peer-relative, with no look-ahead (see [src/judge_workload.py](src/judge_workload.py)):

- **`open_cases_at_filing`** — cases the same judge had open on the focal case's filing date.
- **`aged_open_cases_at_filing`** — of those, how many are "old" for their case family (age past the 75th-percentile completed-case duration).
- **`clearance_rate_last_180_days`** — cases closed ÷ cases newly assigned in the prior 180 days (>1 = docket shrinking, <1 = growing).

## Data

The pipeline starts from a raw federal docket **`Event Log.csv`** (~2.7 GB, not distributed with the repository) and aggregates it to one row per case. Cases are kept only if closed; multi-judge and ambiguous rows are dropped. The build is done in [notebooks/00_eda.ipynb](notebooks/00_eda.ipynb) and produces **`data/by_case.parquet`**, the single input all model code reads. Raw data and the parquet are git-ignored.

To guard against temporal leakage, all splits are **chronological**: models train on the oldest 80% of cases by filing date and are tested on the most recent 20%; hyperparameter tuning uses `TimeSeriesSplit`.

## Models

- **Ridge regression** — linear baseline; L2 penalty tuned per (case type × model level) via `RidgeCV`. Coefficients (on standardized features) give interpretable effects.
- **XGBoost** — gradient-boosted trees; hyperparameters tuned by randomized search, importances read via **SHAP**.

---

## Setup

Requires Python 3.11+. From the project root:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

Place `Event Log.csv` at the project root, then run [notebooks/00_eda.ipynb](notebooks/00_eda.ipynb) end-to-end to generate `data/by_case.parquet`. (If you already have the parquet, skip this step.)

> All commands use `.venv/bin/python3` — the system Python will not have `xgboost`, `shap`, or `torch`.

## Usage

```bash
# 1. Tune XGBoost hyperparameters (writes docs/xgb_best_params_ed.json, reused below)
.venv/bin/python3 scripts/tune_xgb.py --n-iter 20

# 2. Train and compare the four models
.venv/bin/python3 scripts/run_xgb.py                    # both case types
.venv/bin/python3 scripts/run_xgb.py --case-type cv     # civil only
.venv/bin/python3 scripts/run_ridge_regression.py       # alpha auto-tuned (or pass --alpha)

# 3. Cross-model comparison figures
.venv/bin/python3 scripts/compare_ridge_xgb.py
```

Each run computes the judge-workload features on the full dataset first (~30–60 s), then trains. Results are written to `docs/` as JSON and figures to `reports/figures/` as PNG.

### Outputs

| Location | Contents |
| -------- | -------- |
| `docs/*_results_ed.json` | per-model MAE, R², feature lists, SHAP / coefficient importances |
| `docs/xgb_best_params_ed.json` | tuned XGBoost hyperparameters |
| `reports/figures/` | MAE/R² comparisons, SHAP beeswarms, actual-vs-predicted scatter, poster visuals |

## Repository layout

```
src/
  preprocessing.py     # chronological split, the four-model feature contract
  judge_workload.py    # point-in-time judge workload features
scripts/
  tune_xgb.py          # XGBoost hyperparameter search
  run_xgb.py           # XGBoost — 4-model comparison
  run_ridge_regression.py  # Ridge — 4-model comparison
  compare_ridge_xgb.py # cross-model figures
notebooks/
  00_eda.ipynb         # raw Event Log.csv -> data/by_case.parquet
docs/                  # model results (JSON, committed)
reports/figures/       # generated figures
```

See [CLAUDE.md](CLAUDE.md) for deeper architecture notes.

---

## Key finding

Across both Ridge and XGBoost, on civil and criminal cases, **adding dynamic judge workload does not improve prediction of procedural complexity** — Model 2 and Model 4 are statistically indistinguishable from Models 1 and 3. The workload features have ample variance but near-zero correlation with `ed`, so this is a genuine null result rather than a degenerate feature. Procedural complexity is driven overwhelmingly by static case characteristics — most strongly the number of defense counsel and the nature of suit — not by how busy the assigned judge is.

## Acknowledgements

This research was carried out at **Tel Aviv University** under the supervision of **Dr. Shany Azaria**, whose guidance shaped the research question and methodology.
