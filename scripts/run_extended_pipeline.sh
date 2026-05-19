#!/usr/bin/env bash
# Full pipeline including survival and judge workload
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
pip install -q -r requirements.txt

echo "=== 1. Build features (all cv cases, incl. open/censored) ==="
python src/build_features.py --case-type cv

echo "=== 2. Concurrent judge workload ==="
python src/add_judge_workload.py

echo "=== 3. Survival (KM + Cox) ==="
MPLCONFIGDIR=/tmp/mpl python scripts/run_survival.py

echo "=== 4. EDA / regression / ML (closed cases subset in scripts) ==="
MPLCONFIGDIR=/tmp/mpl python scripts/run_eda.py
python scripts/run_regression.py
python scripts/run_ml.py
python scripts/run_process_mining.py --sample-cases 5000

echo "Done. See docs/step7_survival.json and reports/figures/09_*"
