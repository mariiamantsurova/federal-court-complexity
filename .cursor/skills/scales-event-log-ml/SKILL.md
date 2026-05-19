---
name: scales-event-log-ml
description: >-
  Guides EDA, feature engineering, and modeling on the SCALES federal court
  Event Log (ucid, Activity, 61 attribute_* flags, case metadata). Use when
  working on this university project, Event Log.csv, case outcome/duration
  prediction, process mining, or token-efficient legal analytics answers.
---

# SCALES Event Log — University ML Project

## Token rules (always)

1. **Never** load or paste `Event Log.csv` into chat (2.7 GB, ~4.8M rows). Run `python scripts/profile_event_log.py` or chunked code; report numbers only.
2. Answer in **structured bullets/tables**, not prose essays. Skip textbook ML definitions.
3. Default analysis unit: **case (`ucid`)**, not raw event row — unless task is sequence/next-event.
4. Split **cv** vs **cr** early; do not pool without justification (~85% cv).
5. Cite column names from [reference.md](reference.md); read that file only when schema detail is needed.

## Dataset snapshot (verified)

| Metric | Value |
|--------|-------|
| Rows | 4,811,483 |
| Cases (`ucid`) | 175,268 |
| Date range | 1991-01-08 → 2021-07-11 |
| Columns | 98 (61 `attribute_*`) |
| Avg events/case | 27.5 |
| case_type | cv 4,094,131 rows; cr 717,352 |
| Top activities | minute_entry, motion, notice, order, response, complaint |

Data source: [SCALES OKN Event Log](https://docs.scales-okn.org/eventlog/). Activities = broad type; `attribute_*` = subtype/flags (e.g. dispositive, motion to dismiss).

## Course objective (from הצעת פרויקט .pdf)

**Topic**: Legal process complexity vs operational efficiency (Process Mining / Operations Management).

**Main question**: Is there a link between case complexity and procedural efficiency, and does it vary by judge?

**Primary target**: **LOS** (Length of Stay) = days from case open → close per `ucid`.

**Complexity features** (engineer from event log): n_events, n_activity_types, n_motions, n_parties, process-path length, process variability.

**Efficiency**: LOS + optional progress metrics. **Judge** (`District_Judge` / parsed `event_judge`) as moderator — compare patterns across judges.

**Deliverables**: lit review (process mining + OM), metric definitions, complexity↔efficiency analysis, judge comparison, optional process mining / anomaly detection.

## Project phases (checklist)

```
- [ ] 0. Lit review: process mining + OM in legal systems
- [ ] 1. Environment + Parquet subset; parse judges
- [ ] 2. Define & compute complexity + LOS per ucid (EDA, distributions)
- [ ] 3. Correlation / regression: complexity → LOS; judge effects & interactions
- [ ] 4. ML models (RF/XGBoost) + optional process mining (PM4Py) / anomalies
- [ ] 5. Judge-level operational patterns; bottleneck narrative
- [ ] 6. Report: research Q, metrics, models, limitations
```

## Phase detail

### 0 — Metrics (fixed by PDF)

| Family | Examples |
|--------|----------|
| **Complexity** | event count, unique Activity count, motion count, party counts, sequence length, activity entropy / rework loops |
| **Efficiency (Y)** | LOS days; optional: time-to-first-dispositive, events-per-month |
| **Judge** | caseload per judge, judge fixed effects, complexity×judge interaction |

Use **closed** cases only for LOS; document censoring for open cases.

### 1 — Data prep (required for 2.7 GB)

```bash
pip install pandas pyarrow scikit-learn matplotlib seaborn
python scripts/profile_event_log.py          # full stats → stdout
python scripts/profile_event_log.py --sample 50000  # fast dev set
```

Convert once to Parquet partitioned by `case_type` or year; filter columns to ~20–30 for modeling.

**Parse** `event_judge` (stringified tuple) → `judge_id`, `judge_role`.  
**Dedup**: respect `attribute_duplicates` (mostly 1; some rows repeat same event).

### 2 — Case-level EDA

Per `ucid` aggregate:
- `n_events`, `span_days`, first/last `Activity`
- counts per Activity; any dispositive/settlement attribute fired
- static: `nature_suit`, `city`, party counts, `is_mdl`, judge fields

Plots: distribution of `span_days`; outcome bar chart; nature_suit top-N; cv vs cr comparison.

### 3 — Event-level EDA

- Activity transition matrix (top-k activities)
- Attribute prevalence (% True per column) — expect heavy sparsity
- Timeline: filings per year; check 2020+ covid shift if relevant

### 4 — Features

**Static (case)**: `case_type`, `nature_suit`, `city`, plaintiff/defendant counts & shares, judge id, `is_mdl`, `related_case_count`.

**Temporal (engineered)**:
- First-k events Activity sequence (hashed n-grams)
- Time to first motion / first dispositive event
- Counts: motions, orders, notices; sum of scheduling attributes
- Last activity before close

**Leakage guard**: for prediction at filing, use only events in first N days or first M events.

### 5 — Models (per PDF)

1. **Exploratory**: Pearson/Spearman complexity vs LOS; OLS with judge dummies.
2. **Prediction**: Linear regression → Random Forest / XGBoost; report feature importance (complexity vs judge vs case type).
3. **Judge heterogeneity**: include judge + complexity×judge; compare RMSE/MAE by judge bucket.
4. **Optional**: PM4Py process maps; anomaly detection on trace length / LOS residuals by judge.

**Validation**: group by `ucid`; time split (train older, test newer). **Metrics**: MAE, RMSE, R² on LOS; residual plots by judge.

### 6 — Report essentials

Research question → data scope → target definition → features → model → results table → confusion/errors → limitations (Chicago-heavy sample, label noise, sparse attributes).

## Response templates

**EDA summary**
```
Cases: N | Target: X | Imbalance: ratio
Key findings: (1)… (2)… (3)…
Next: feature list / model baseline
```

**Model recommendation**
```
Target: … | Split: group by ucid + time
Features: static + …
Model: … because …
Metric: …
Risk: leakage from …
```

## Do not

- Train on full 4.8M rows in a notebook without sampling or Parquet.
- Mix cv and cr in one classifier without stratification analysis.
- Use post-dispositive events to predict dispositive outcome.
- Paste raw CSV rows into chat.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/profile_event_log.py` | Streaming profile; optional `--sample N` |

## Extra reference

- Full column list + types: [reference.md](reference.md)
