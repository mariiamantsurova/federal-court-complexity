# Project log — Complexity vs efficiency (SCALES Event Log)

**Research question:** Is case complexity linked to LOS (efficiency), and does this differ by judge?

**Advisor / client:** Dr. Shani Azaria, Tel Aviv University  
**Data:** Northern District of Illinois Event Log (~175k cases, ~4.8M events, 1991–2021)

---

## How to use this log

After each work session, add an entry under the relevant step:
- **Date**
- **What you did** (command, notebook cell, decision)
- **Output** (file path, key number, figure)
- **Decision / rationale** (why you chose this)
- **Open issues** (blockers, questions)

---

## Progress overview

| Step | Status | Artifact |
|------|--------|----------|
| 0. Lit review | 🟡 template | `docs/literature_notes.md` (fill before submit) |
| 1. Environment & data profile | ✅ | `docs/step1_profile_output.txt` |
| 2. Feature build (full data) | ✅ | `data/case_features.parquet` (151,640 cv closed) |
| 3. EDA | ✅ | `reports/figures/`, `docs/step3_eda_summary.json` |
| 4. Regression / judge effects | ✅ | `docs/step4_regression.json`, `notebooks/02_regression.ipynb` |
| 5. ML models | ✅ | `docs/step5_ml.json`, `notebooks/03_ml_models.ipynb` |
| 6. Process mining (optional) | ✅ | `docs/step6_process_mining.json`, `notebooks/04_process_mining.ipynb` |
| 7. Final report | ✅ | `docs/final_report.md` (updated post-implementation) |
| 8. Survival + workload | ✅ | `step7_survival.json`, `add_judge_workload.py` |

---

## Step 0 — Literature review

**Goal:** Process mining + operations management in legal systems; justify complexity & LOS metrics.

**Tasks:**
- [ ] 3–5 papers (process mining in courts, case duration, judge effects)
- [ ] Summarize gaps → motivates your data

**Document here:**

### Session template
```
Date:
Papers read:
Key quotes / definitions (complexity, LOS, efficiency):
How this supports our metrics:
```

---

## Step 1 — Environment & raw data profile

**Goal:** Confirm data scale before modeling.

**Done:**
- [x] Python venv + `requirements.txt`
- [x] `scripts/profile_event_log.py` created

**Done (2026-05-19):**
- [x] Full profile → `docs/step1_profile_output.txt`
- 4,811,483 rows; 175,268 cases; 1991–2021; cv 85% / cr 15%

**Document here:**

```
Date: 2026-05-19
Rows / cases: 4,811,483 / 175,268
date range: 1991-01-08 -> 2021-07-11
case_type: cv 4,094,131 rows; cr 717,352
```

---

## Step 2 — Build case-level features

**Goal:** One row per `ucid` with complexity features + `los_days` (closed cases).

**Metric definitions (for report):**

| Feature | Definition |
|---------|------------|
| `los_days` | Days between first and last event (closed cases only) |
| `n_events` | Total events in case |
| `n_activity_types` | Unique `Activity` values |
| `n_motions` | Count of `Activity == motion` |
| `activity_entropy` | Shannon entropy over activity distribution |
| `rework_ratio` | Share of repeated activity types |
| `time_gaps_std` | Std of days between consecutive event dates |
| `party_load` | plaintiffs_count + Defendants_count |
| `complexity_index` | Mean of z-scored complexity columns |

**Done:**
- [x] `src/build_features.py`
- [x] Sample run: `--sample-rows 200000 --case-type cv --closed-only` → 7,381 cases

**Done (2026-05-19):**
- [x] `python src/build_features.py --case-type cv --closed-only`
- 151,640 cases; median LOS 253 days
- Winsorized complexity at 1–99 pct; `complexity_index` clipped [-3, 3]

**Document here:**

```
Date: 2026-05-19
Output: data/case_features.parquet
N cases: 151,640 (cv, closed)
Median LOS: 253 days
```

---

## Step 3 — Exploratory data analysis

**Goal:** Understand distributions and complexity ↔ LOS relationship.

**Tasks:**
- [ ] Run all cells in `notebooks/01_build_case_features.ipynb`
- [ ] Save figures to `reports/figures/`
- [ ] Correlation table complexity vs `los_days`
- [ ] Compare LOS across `complexity_index` quartiles
- [ ] Note skew → consider `log1p(los_days)` for modeling

**Done (2026-05-19):** `python scripts/run_eda.py` → `reports/figures/01–04_*.png`

**Findings:**
- 151,640 cases; median LOS **253** days (mean 425)
- Strongest correlate: **complexity_index r=0.65**
- LOS by complexity quartile (median days): Q1 **69**, Q2 **172**, Q3 **346**, Q4 **691**

**Document here:**

```
Date: 2026-05-19
Figures: reports/figures/01-04
Key insight: higher complexity → much longer LOS (monotone across quartiles)
```

---

## Step 4 — Regression & judge effects

**Goal:** Test main research question with interpretable models.

**Models (in order):**
1. `los_days ~ complexity_index + nature_suit`
2. Add `District_Judge` fixed effects
3. Interaction: `complexity_index * District_Judge`

**Metrics:** R², MAE, RMSE; residual plots by judge.

**Done (2026-05-19):** `python scripts/run_regression.py` (120,940 train / 30,236 test; judges ≥200 cases)

| Model | R² (log LOS) | MAE (days) |
|-------|----------------|------------|
| M1 complexity + suit | 0.515 | 305 |
| M2 + judge FE | 0.536 | 298 |
| M3 + complexity×judge | 0.547 | 289 |

**Interpretation:** Complexity predicts longer cases (r=0.65). Adding judges improves fit (+0.02 R² log). Interaction model best — slope may differ by judge.

**Document here:**

```
Date: 2026-05-19
See: docs/step4_regression.json
Next: Random Forest (03) + write report section
```

---

## Step 5 — Machine learning

**Goal:** Predict LOS; rank feature importance (complexity vs judge vs controls).

**Done (2026-05-19):** `python scripts/run_ml.py`

| Model | R² (log) | MAE (days) | Note |
|-------|----------|------------|------|
| RF full | 0.945 | 76 | Includes `n_events`, `time_gaps_std` (mechanical with LOS) |
| RF restricted | see `step5_ml.json` | — | Excludes volume/timing; fairer interpretation |
| Regression M3 | 0.547 | 289 | Linear benchmark |

**Importance (full RF):** complexity_index, time_gaps_std, n_events dominate. Judge/suit &lt;0.2% combined.

**Caveat for report:** High RF R² partly reflects circular features (more events → longer case). Use restricted model + regression for conclusions.

**Document here:**

```
Date: 2026-05-19
Figure: reports/figures/06_rf_feature_importance.png
See: docs/step5_ml.json
```

---

## Step 6 — Process mining & anomalies (optional)

**Goal:** Secondary PDF questions — variants, bottlenecks, outliers by judge.

**Done (2026-05-19):** `python scripts/run_process_mining.py --sample-cases 5000`

**Process mining (5,000 cv cases sampled):**
- 4,430 unique variants — high heterogeneity
- Most common path: `complaint > motion > order` (33 cases in sample)
- Dominant transitions: `minute_entry→minute_entry`, `motion→notice`, `notice→minute_entry`
- **Bottleneck:** motion → next order, median **50 days** (p90 **306 days**)

**Anomalies (full 151k cases, top 1% residual after complexity+judge+suit):**
- 1,512 cases; median excess LOS **~1,186 days** vs prediction
- Some extreme outliers have *low* complexity but very long LOS (stalled cases)
- Highest anomaly rates by judge: SJ000237 (3.2%), SJ000212 (2.2%) — see `07_anomaly_rate_by_judge.png`

**Artifacts:** `docs/step6_process_mining.json`, figures `07_*`, `08_*`, `notebooks/04_process_mining.ipynb`

**Document here:**

```
Date: 2026-05-19
Tool: pandas stream mining (no PM4Py)
Variant: complaint→motion→order most frequent short path
Bottleneck: 50d median motion→order
Anomaly: 1% residual; clustered on some judges; long LOS not always high complexity
```

---

## Step 7 — Final report

**Sections:**
1. Introduction & research question  
2. Literature review  
3. Data & methods (metrics table from Step 2)  
4. EDA results  
5. Models & judge comparison  
6. Discussion & limitations  
7. Conclusion  

**Limitations included:** single district, observational design, LOS definition, mechanical features, judge anonymization.

---

## Session log (chronological)

### 2026-05-19
- **Step:** 1–4 pipeline
- **Done:** Full profile, feature build, EDA figures, regression baselines
- **Output:** `case_features.parquet`, `reports/figures/`, `docs/step*_*.json/txt`
- **Next:** Final report (Step 7)

### 2026-05-19 (Step 6)
- **Step:** Process mining + anomalies
- **Done:** Variants, transitions, motion→order bottleneck, LOS anomaly detection
- **Output:** `step6_process_mining.json`, figures 07–08
- **Next:** Write final report

### 2026-05-19 (Step 7)
- **Step:** Final report
- **Done:** `docs/final_report.md` draft
- **Next:** Lit review + export PDF for submission

### YYYY-MM-DD
- **Step:**
- **Done:**
- **Output:**
- **Next:**
