# Federal Court Complexity vs Efficiency

University project: modeling the relationship between **case complexity** and **operational efficiency (LOS)** using the SCALES federal court event log (Northern District of Illinois).

## Setup

1. Place `Event Log.csv` in the project root (not in git — file is ~2.7 GB).
2. Create venv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the full pipeline:

```bash
bash scripts/run_extended_pipeline.sh
```

## Outputs

- `data/case_features.parquet` — case-level features (generated)
- `reports/figures/` — EDA and model figures
- `docs/final_report.md` — project report

## Structure

| Path | Purpose |
|------|---------|
| `src/build_features.py` | Stream CSV → case features |
| `src/add_judge_workload.py` | Concurrent judge caseload |
| `scripts/run_*.py` | EDA, regression, ML, survival, process mining |
| `notebooks/` | Jupyter notebooks |
| `docs/` | Report, log, results JSON |
