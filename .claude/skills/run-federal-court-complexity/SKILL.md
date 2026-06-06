---
name: run-federal-court-complexity
description: Run, build, test, or smoke-test the federal court complexity pipeline. Use when asked to run scripts, train models, run the pipeline, test preprocessing, or verify results.
---

# Federal Court Complexity Pipeline

Python ML pipeline modeling case complexity vs. LOS (Length of Stay) in the Northern District of Illinois federal courts.

**Stack:** pandas, scikit-learn, XGBoost, SHAP, statsmodels, PyTorch, sentence-transformers, pm4py  
**Venv:** `.venv/` at project root — always use `.venv/bin/python3`  
**Data:** Pre-built parquet files live in `data/` (no raw CSV needed for most tasks)

---

## Prerequisites

- Python 3.11 (via Anaconda at `/opt/anaconda3`) — system Python lacks xgboost/shap
- `.venv/` already exists and is fully installed
- `data/case_features.parquet` (168 k rows) — already present, not in git
- `data/aggregations/by_case.parquet`, `by_judge.parquet` — already present
- `Event Log.csv` (~2.7 GB) — required **only** for `scripts/build_features.py` (Step 1 rebuild)

---

## Setup (one time, already done)

```bash
cd "/Users/mariamantsurova/Desktop/University/3rd Year/federal-court-complexity"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## What scripts actually exist

The README references `scripts/run_pipeline.sh` — **not present** (no end-to-end shell runner exists yet).

Scripts that **exist and work**:

| Path                              | Purpose                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| `scripts/build_features.py`       | Step 1: `Event Log.csv` → `Event Log_model.csv` (needs 2.7 GB CSV)                  |
| `scripts/run_rf_shap.py`          | **RF + SHAP** — trains Random Forest, produces importance CSV + SHAP bar plot        |
| `scripts/run_xgb_shap.py`        | **XGBoost + SHAP** — trains XGBoost, produces beeswarm + bar SHAP plots              |
| `scripts/run_neural_net.py`       | **Neural Net** — custom learned embeddings or HF sentence-transformer judge embeddings |
| `src/build_case_features.py`      | Step 2: events → `data/case_features.parquet` (needs `Event Log_model.csv`)         |
| `src/build_aggregations.py`       | Step 3: cases → `data/aggregations/` parquets                                        |
| `src/preprocessing_trees.py`      | Feature matrix builder for RF / XGBoost                                              |
| `src/preprocessing_neural_net.py` | Feature matrix + judge/suit encoding for neural net                                  |
| `src/neural_net_model.py`         | PyTorch `CourtCaseNeuralNet` and `CourtCaseNeuralNetLSTM` modules                    |
| `src/suit_features.py`            | Suit-type feature engineering                                                        |
| `src/judge_vocabulary.py`         | Judge name → integer embedding index                                                 |

**Note:** `src/features.py` is imported by `build_aggregations.py` but does not exist — that script will error on import. Use `preprocessing_trees.py` directly instead.

---

## Run the pipeline (agent path)

### Random Forest + SHAP

```bash
cd "/Users/mariamantsurova/Desktop/University/3rd Year/federal-court-complexity"

# All cases (pooled)
.venv/bin/python3 scripts/run_rf_shap.py

# Civil only
.venv/bin/python3 scripts/run_rf_shap.py --case-type cv

# Criminal only, excluding MDL cases
.venv/bin/python3 scripts/run_rf_shap.py --case-type cr --exclude-mdl
```

Outputs: `docs/01_ml_results{suffix}.json`, `docs/01_feature_importance{suffix}.csv`, `reports/figures/01_rf_feature_importance{suffix}.png`

Verified results (pooled, n_estimators=200, 2026-06-05):
```
MAE=239.9  RMSE=473.2  R²=0.4698
Top SHAP: n_events, complexity_index, sum_attribute_hearing_conf, n_motions, n_activity_types
```

### XGBoost + SHAP

```bash
# All cases
.venv/bin/python3 scripts/run_xgb_shap.py

# Civil only
.venv/bin/python3 scripts/run_xgb_shap.py --case-type cv

# With MDL exclusion
.venv/bin/python3 scripts/run_xgb_shap.py --exclude-mdl
```

Outputs: `docs/02_xgb_shap_results{suffix}.json`, `docs/02_xgb_shap_importance{suffix}.csv`, `reports/figures/02_xgb_shap_bar{suffix}.png`, `reports/figures/02_xgb_shap_beeswarm{suffix}.png`

Verified results (pooled, n_estimators=500, 2026-06-05):
```
MAE=242.9  RMSE=475.0  R²=0.4657
Top SHAP: n_events, sum_attribute_hearing_conf, complexity_index, Defendants_share_ind, Defendants_pending_counts
```

### Neural Network

Two judge-representation modes, otherwise identical architecture:

**Custom (learned embedding)** — judge IDs embedded from scratch:
```bash
.venv/bin/python3 scripts/run_neural_net.py
.venv/bin/python3 scripts/run_neural_net.py --case-type cv
```

**HuggingFace (sentence-transformers/all-MiniLM-L6-v2)** — pretrained 384-dim judge name embeddings, projected to 16-dim via trainable linear layer, then frozen:
```bash
.venv/bin/python3 scripts/run_neural_net.py --use-hf-embeddings
.venv/bin/python3 scripts/run_neural_net.py --use-hf-embeddings --case-type cv
```

Key flags:
```
--epochs N          (default 40)
--batch-size N      (default 1024; use 2048 for speed on MPS/GPU)
--lr FLOAT          (default 1e-3)
--hidden-dim N      (default 128)
--embedding-dim N   (default 16, judge embedding output dim)
--patience N        (default 8, early stopping)
```

Outputs: `docs/04_neural_net_results{suffix}.json`, `docs/04_nn_model{suffix}.pt`, `reports/figures/04_nn_loss_curve{suffix}.png`

Verified results (pooled, 40 epochs, batch 2048, MPS, 2026-06-05):
```
Custom embeddings:  MAE=247.5  RMSE=522.4  R²=0.3417  (54s)
HF embeddings:      MAE=246.3  RMSE=519.8  R²=0.3482  (62s)
```
Note: model still improving at epoch 40; run with `--epochs 80` for better convergence. Tree models outperform NN on this tabular dataset (expected).

### Run Step 3 — rebuild aggregations (needs case_features.parquet)

```bash
# build_aggregations.py imports 'features' module which doesn't exist.
# When features.py is created, run:
# .venv/bin/python3 src/build_aggregations.py
```

### Run Step 1 (only if Event Log.csv is present)

```bash
ls -lh "Event Log.csv"   # check it's there (~2.7 GB)
.venv/bin/python3 scripts/build_features.py
```

---

## Run (human path)

```bash
source .venv/bin/activate
jupyter notebook notebooks/00_eda.ipynb   # EDA
jupyter notebook notebooks/01_aggregated.ipynb
```

---

## Direct module invocation

Import and call internal code without running the full pipeline:

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.preprocessing_trees import prepare_for_trees
import pandas as pd
df = pd.read_parquet('data/aggregations/by_case.parquet')
X, y = prepare_for_trees(df)
print(X.shape, X.columns.tolist()[:5])
"
```

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.neural_net_model import CourtCaseNeuralNet
import torch
m = CourtCaseNeuralNet(n_numeric_features=30, judge_vocab_size=100, suit_vocab_size=50)
print(m)
"
```

---

## Preprocessing summary

### Random Forest + SHAP and XGBoost + SHAP

Both use `src/preprocessing_trees.py` → `prepare_for_trees(df)`.  
Input: `data/aggregations/by_case.parquet`. Output: `(X, y)` — 168 k rows × ~35 features, target = `los_days`.

**Feature groups included:**

| Group                   | Features                                                                                                                                                                                                                                                                                                        | Notes                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Complexity core         | `n_events`, `n_activity_types`, `n_motions`, `activity_entropy`, `complexity_index`                                                                                                                                                                                                                             | Always included                                              |
| Case / party attributes | `plaintiffs_count`, `plaintiffs_share_ind/pro_se/pro_hac_vice`, `plaintiffs_counsels_count`, `Defendants_count`, `Defendants_share_ind/pro_se/pro_hac_vice`, `Defendants_counsels_count`, `Defendants_pending_counts`, `Defendants_terminated_counts`, `Other_courts`, `related_case_count`, `Magistrate_Judge` | Numeric; `Magistrate_Judge` treated as boolean integer       |
| Party type counts       | `Party_Amicus`, `Party_Counter_Claimant`, `Party_Counter_Defendant`, `Party_Court_Monitor`, `Party_Intervenor`, `Party_Material_Witness`, `Party_Third_Party_Defendant`, `Party_Third_Party_Plaintiff`, `Party_Trustee`                                                                                         | All numeric                                                  |
| Event-type aggregations | `sum_attribute_scheduling`, `sum_attribute_hearing_conf`, `sum_attribute_dismissal_other`, `sum_attribute_dispositive`, `sum_attribute_opening`                                                                                                                                                                 | Sparse counts                                                |
| Suit structure          | `n_unique_suits`, `suit_entropy`, `is_multisuit`, `suit_dominance`, `has_<suittype>`, `suit_freq_<suittype>`                                                                                                                                                                                                    | Derived from `nature_suits` array via `src/suit_features.py` |
| City                    | One-hot encoded (drop first)                                                                                                                                                                                                                                                                                    | `city` column                                                |

**Excluded:** `ucid`, `case_open_date`, `case_type`, `District_Judge`, `is_mdl`, `los_days`, `log_los_days`, `nature_suit`, `nature_suits`

**Imputation:** median for all numeric columns (via `SimpleImputer`)  
**Scaling:** none by default (trees are scale-invariant); `scale=True` applies `StandardScaler`  
**SHAP:** `shap.TreeExplainer` — works natively with both RF and XGBoost; produces per-feature attributions on the same feature matrix

---

### Neural Network (`CourtCaseNeuralNet`)

Uses `src/preprocessing_neural_net.py` → `prepare_for_neural_net(df)`.  
Returns a dict of tensors (not a single DataFrame) + `y`.

**Three input streams:**

| Stream           | Source                                                                                         | Processing                                                                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Numeric features | Same complexity + case attributes + party types as trees (no city, no suit structure features) | Median imputation → `StandardScaler` → `float32` array of shape `(n_cases, 29)`                                                                      |
| Judge embedding  | `District_Judge` column                                                                        | `JudgeVocabulary` encodes each judge name to an integer ID (vocab = 93 judges + `<UNK>`); stored as `(n_cases,)` int array                           |
| Suit embedding   | `nature_suits` array column                                                                    | `SuitVocabulary` encodes each suit type to an integer; sequences padded/truncated to `max_suit_length=20`; stored as `(n_cases, 20)` + a binary mask |

**Key difference from trees:** no one-hot encoding — `District_Judge` and suit types go into learned embedding layers, not dummy columns.

**Architecture (`CourtCaseNeuralNet`):**

```
numeric (29) ──────────────────────────────┐
judge_id (1) → Embedding(93, 16) ──────────┼─→ cat → FC(→128) → BN → ReLU → Dropout(0.3)
suits (20,)  → Embedding(vocab,16)          │                  → FC(→64)  → BN → ReLU → Dropout
               + masked mean-pool → (16) ──┘                  → FC(→32)  → BN → ReLU → Dropout
                                                               → FC(→1)   = LOS prediction
```

Variant `CourtCaseNeuralNetLSTM` replaces masked mean-pooling over suit embeddings with a 2-layer LSTM, using the final hidden state as the suit representation.

**Invoke preprocessing:**

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.preprocessing_neural_net import prepare_for_neural_net
df = pd.read_parquet('data/aggregations/by_case.parquet')
data, y = prepare_for_neural_net(df)
print('numeric:', data['numeric_features'].shape)
print('judges vocab size:', data['judges_vocab_size'])
print('suits encoded:', data['suits_encoded'].shape)
"
```

Expected:

```
numeric: (168107, 29)
judges vocab size: 93
suits encoded: (168107, 20)
```

---

## Gotchas

- **`features.py` is missing.** `src/build_aggregations.py` does `from features import ...` but no such file exists. The script will crash on import. Use `preprocessing_trees.py` directly instead.
- **`sys.path.insert(0, '.')` is required.** Running `src/build_aggregations.py` or any `src/` script needs the project root on the path. The scripts do this themselves; when importing in a `-c` snippet, add it manually.
- **`scripts/run_pipeline.sh` does not exist.** The README describes a full pipeline runner that has not been written yet.
- **`.venv/bin/python3` is the only working interpreter.** System Python at `/opt/anaconda3/bin/python3` lacks xgboost and shap.
- **`data/aggregations/by_case.parquet` has `complexity_index`.** `data/case_features.parquet` does not — it's added by the aggregation step. Always use `by_case.parquet` as the model input.
- **Import path for `src/suit_features.py`:** imported as `from src.suit_features import ...` (with `src.` prefix) when running from project root.

---

## Troubleshooting

| Error                                                      | Fix                                                                        |
| ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'features'`          | `features.py` missing — use `preprocessing_trees.py` directly              |
| `ModuleNotFoundError: No module named 'xgboost'`           | Use `.venv/bin/python3`, not system python3                                |
| `FileNotFoundError: data/aggregations/by_case.parquet`     | Run `src/build_aggregations.py` first (needs `data/case_features.parquet`) |
| `FileNotFoundError: Event Log.csv`                         | Raw event log not in repo — needed only for Step 1                         |
| `ModuleNotFoundError: No module named 'src.suit_features'` | Add `sys.path.insert(0, str(ROOT))` or run from project root               |
