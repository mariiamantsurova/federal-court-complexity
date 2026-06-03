# Survival analysis — step-by-step guide

Survival analysis models **time until case closure**, keeping **open cases** as **censored** observations (instead of dropping them).

| Column | Meaning |
|--------|---------|
| `duration_days` | Days from first to last event in the log |
| `event_observed` | `1` = case closed, `0` = still open (censored) |

**Hazard ratio (Cox) > 1** → higher complexity is linked to **faster closure** (higher hazard).  
**HR < 1** → linked to **slower** closure (case stays open longer).

---

## Step 1 — Build the survival dataset

Reads **raw** `Event Log.csv` (open + closed cases).

```bash
source .venv/bin/activate

# Full run (~5–10 min)
python src/build_survival_dataset.py

# Quick test
python src/build_survival_dataset.py --sample-rows 500000
```

**Output:** `data/survival_cases.parquet`

Check: printed counts of closed vs censored cases.

---

## Step 2 — Run Kaplan–Meier + Cox (pooled)

```bash
python scripts/run_survival_analysis.py
```

**Outputs:**
- `reports/figures/04_km_by_case_type.png` — survival curves cv vs cr
- `reports/figures/04_km_by_complexity.png` — curves by complexity quartile
- `docs/04_survival_results.json` — summary + Cox coefficients
- `docs/tables/T6_survival_cox.csv` — Cox table for report

---

## Step 3 — Run by case type (civil / criminal)

```bash
python scripts/run_survival_analysis.py --case-type cv
python scripts/run_survival_analysis.py --case-type cr
```

**Outputs:** `*_cv.json`, `*_cr.json`, `T6_survival_cox_cv.csv`, etc.

---

## Step 4 — Interpret for the report

1. **Censoring rate** — what % of cases were still open at data cutoff?
2. **KM curves** — do civil cases close faster than criminal? Do high-complexity cases stay open longer?
3. **Cox M1** — which complexity feature affects hazard of closure? Compare to OLS M3 on closed cases only.
4. **Limitation** — censoring date is end of dataset (2021), not “today”.

---

## How this relates to your other models

| Method | Cases | Target |
|--------|-------|--------|
| RF / XGBoost | Closed only | Predict `los_days` |
| Causal OLS | Closed only | Explain log(LOS) |
| **Survival (Cox/KM)** | **Open + closed** | Time to closure with censoring |
