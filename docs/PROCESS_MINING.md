# Process mining — step-by-step (for your report)

This document explains how the process-mining figures were built and how to describe them in the thesis.

## What is process mining here?

Each **case** (`ucid`) is a **trace**: a time-ordered sequence of **events** (`Activity` at `date_filed`).
Process mining discovers **patterns in those sequences** — not just averages like LOS or `n_events`.

---

## Step 1 — Define the cohort

| Rule | Value |
|------|--------|
| Case type | Civil (`cv`) only |
| MDL | Excluded (`is_mdl != True`) |
| Status | Closed cases (in `Event Log_model.csv`) |
| Trace length | Between 10 and 80 events (readable plots) |

Population for selection: **91,845** civil non-MDL closed cases with valid LOS.

---

## Step 2 — Select illustrative traces (Figure: `05_trace_exemplars_cv.png`)

Four **archetypes** (one case each) — not random samples:

| Case | Archetype | Purpose |
|------|-----------|---------|
| A | Typical duration (~median LOS) | LOS=300 d, events=26, complexity=0.19 |
| B | Long duration (~90th pct LOS) | LOS=1100 d, events=28, complexity=-0.08 |
| C | High complexity, mid LOS | LOS=443 d, events=76, complexity=3.03 |
| D | Mid complexity, very long LOS | LOS=6573 d, events=23, complexity=-0.10 |

**How to read the plot:**
- **X-axis:** days since the first event in the case.
- **Y-axis:** event order (1st, 2nd, 3rd, …).
- **Color:** activity type (`motion`, `order`, `notice`, …).

**Report sentence (template):**
> Figure X shows four illustrative civil traces. Case B (long LOS) exhibits more mid-process 
> motion–response activity than Case A (median LOS), while Case C separates high procedural 
> complexity from extreme duration (Case D).

Full case metadata: `docs/tables/T7_trace_exemplars.csv` (appendix; anonymize `ucid` if required).

---

## Step 3 — Discover the directly-follows graph (Figure: `05_dfg_cv_sample.png`)

1. Random sample of **800** cases from the same cohort.
2. Build a standard event log: case id = `ucid`, activity = `Activity`, timestamp = `date_filed`.
3. Run **DFG discovery** with PM4Py: edge *A → B* = how often activity B immediately follows A.
4. Keep edges with frequency ≥ **30**; render with matplotlib (arrow width ∝ count).

**How to read the DFG:**
- **Nodes** = activities; **thick arrows** = frequent direct successors.
- **Start/end** activities show how cases open and close in the sample.

**Report sentence (template):**
> The directly-follows graph on a stratified civil sample shows that `motion` and `minute_entry` 
> are the most frequent hubs; dispositive paths often pass through `order` after `motion` or `response`.

**Top edges in sample:**

| From | To | Count |
|------|-----|-------|
| minute_entry | minute_entry | 3249 |
| notice | minute_entry | 1974 |
| motion | notice | 1973 |
| minute_entry | motion | 1348 |
| minute_entry | order | 785 |
| notice | motion | 516 |
| order | minute_entry | 516 |
| motion | minute_entry | 461 |
| motion | order | 375 |
| complaint | summons | 374 |

---

## Step 4 — Compare low vs high LOS transitions (Figure: `05_transitions_cv_q1_q4.png`)

1. Split cohort into LOS **quartiles**.
2. Sample **500** cases from Q1 (short LOS) and **500** from Q4 (long LOS).
3. Count consecutive activity pairs; keep top **10** activities.
4. **Row-normalize** each heatmap → cell = P(next activity | current activity).

**How to read the heatmaps:**
- Compare the same row (e.g. `motion`) across panels: where probability mass shifts is the structural difference.
- Higher values on `motion`→`response` or `motion`→`order` in Q4 support “more procedural churn” on long cases.

**Report sentence (template):**
> Compared with Q1, Q4 traces allocate a larger share of transitions from `motion` to follow-up 
> activities rather than early disposition, consistent with longer LOS.

---

## Step 5 — Link back to your main models

| Process-mining view | Your existing metrics |
|--------------------|------------------------|
| More loops / branches in DFG & heatmaps | Higher `n_activity_types`, `activity_entropy` |
| More `motion` transitions | Higher `n_motions` |
| Longer spans between events | Higher `los_days` even at moderate complexity |

State clearly: this is **descriptive** process structure, not causal identification.

---

## Step 6 — Limitations (include in report)

- Sample-based DFG and heatmaps (not full 4.8M-row log).
- Civil non-MDL only; criminal paths differ.
- Activities are coarse (`motion`, `order`); fine-grained `attribute_*` not shown.
- Illustrative cases are cherry-picked for clarity, not population averages.

---

## Files

| File | Role |
|------|------|
| `reports/figures/05_trace_exemplars_cv.png` | Four case timelines |
| `reports/figures/05_dfg_cv_sample.png` | PM4Py directly-follows graph |
| `reports/figures/05_transitions_cv_q1_vs_q4.png` | LOS Q1 vs Q4 transitions |
| `docs/tables/T7_trace_exemplars.csv` | Case-level appendix table |
| `docs/05_process_mining.json` | Run metadata |

## Regenerate

```bash
python scripts/run_process_mining_viz.py
```
