#!/usr/bin/env bash
# Full pipeline: raw event log → aggregations → ML / SHAP / causal regression.
#
# Usage:
#   bash scripts/run_pipeline.sh                    # full run (pooled models)
#   bash scripts/run_pipeline.sh --by-case-type     # also cv + cr models
#   bash scripts/run_pipeline.sh --sample           # dev: 500k event rows
#   bash scripts/run_pipeline.sh --skip-clean       # reuse Event Log_model.csv
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

SAMPLE_ROWS=""
SKIP_CLEAN=0
SKIP_CASE=0
SKIP_MODELS=0
BY_CASE_TYPE=0

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \?//'
  echo ""
  echo "Options:"
  echo "  --sample          Use 500,000 event rows (fast dev run)"
  echo "  --by-case-type    Build cv/cr aggregations and run models for civil + criminal"
  echo "  --skip-clean      Skip scripts/build_features.py (reuse Event Log_model.csv)"
  echo "  --skip-case       Skip src/build_case_features.py (reuse data/case_features.parquet)"
  echo "  --skip-models     Build data only; skip ML / SHAP / causal scripts"
  echo "  -h, --help        Show this help"
}

for arg in "$@"; do
  case "$arg" in
    --sample) SAMPLE_ROWS=500000 ;;
    --by-case-type) BY_CASE_TYPE=1 ;;
    --skip-clean) SKIP_CLEAN=1 ;;
    --skip-case) SKIP_CASE=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
  esac
done

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# bash 3.2 + set -u: expanding empty arrays ("${arr[@]}") is an error
run_with_sample() {
  if [[ -n "$SAMPLE_ROWS" ]]; then
    "$@" --sample-rows "$SAMPLE_ROWS"
  else
    "$@"
  fi
}

run_models() {
  local case_type="$1"
  if [[ -n "$case_type" ]]; then
    log "Models [$case_type]: RF/DT + XGBoost/SHAP + causal regression"
    "$PYTHON" scripts/run_ml_importance.py --features "${AGG_CASE}" --case-type "$case_type"
    "$PYTHON" scripts/run_xgb_shap.py --features "${AGG_CASE}" --case-type "$case_type"
    "$PYTHON" scripts/run_causal_regression.py --features "${AGG_CASE}" --case-type "$case_type"
  else
    log "Models [all]: RF/DT + XGBoost/SHAP + causal regression"
    "$PYTHON" scripts/run_ml_importance.py --features "${AGG_CASE}"
    "$PYTHON" scripts/run_xgb_shap.py --features "${AGG_CASE}"
    "$PYTHON" scripts/run_causal_regression.py --features "${AGG_CASE}"
  fi
}

# --- environment ---
if [[ ! -x "$PYTHON" ]]; then
  log "Creating venv and installing requirements..."
  python3 -m venv "$VENV"
  "$PIP" install -r requirements.txt
fi

if [[ "$(uname -s)" == "Darwin" ]] && ! "$PYTHON" -c "import xgboost" 2>/dev/null; then
  if command -v brew >/dev/null 2>&1 && ! brew list libomp &>/dev/null; then
    log "Installing libomp (required for XGBoost on macOS)..."
    brew install libomp
  fi
  "$PIP" install -q xgboost shap
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER:-user}}"

RAW="${ROOT}/Event Log.csv"
MODEL_CSV="${ROOT}/Event Log_model.csv"
CASE_PARQUET="${ROOT}/data/case_features.parquet"
AGG_DIR="${ROOT}/data/aggregations"
AGG_CASE="${AGG_DIR}/by_case.parquet"

if [[ -n "$SAMPLE_ROWS" ]]; then
  log "Dev mode: sampling ${SAMPLE_ROWS} event rows"
fi

# --- step 1: clean event log (closed cases only) ---
if [[ "$SKIP_CLEAN" -eq 1 ]]; then
  log "Skipping event cleaning (--skip-clean)"
elif [[ -f "$MODEL_CSV" && -z "$SAMPLE_ROWS" ]]; then
  log "Reusing existing Event Log_model.csv (delete it or omit --skip-clean to rebuild)"
  python3 - <<'PY' "$MODEL_CSV"
import sys
from pathlib import Path
p = Path(sys.argv[1])
print(f"  {p} ({p.stat().st_size / 1e9:.2f} GB)")
PY
else
  log "Step 1/6: Clean event log → Event Log_model.csv (closed cases only)"
  [[ -f "$RAW" ]] || { echo "Missing: $RAW" >&2; exit 1; }
  run_with_sample "$PYTHON" scripts/build_features.py
fi

# --- step 2: case-level features ---
if [[ "$SKIP_CASE" -eq 1 ]]; then
  log "Skipping case aggregation (--skip-case)"
else
  log "Step 2/6: Aggregate events → data/case_features.parquet"
  run_with_sample "$PYTHON" src/build_case_features.py --input "$MODEL_CSV"
fi

# --- step 3: judge / city / case aggregations ---
log "Step 3/6: Build aggregations (by_case, by_judge, by_city)"
if [[ "$BY_CASE_TYPE" -eq 1 ]]; then
  "$PYTHON" src/build_aggregations.py --input "$CASE_PARQUET" --by-case-type
else
  "$PYTHON" src/build_aggregations.py --input "$CASE_PARQUET"
fi

if [[ "$SKIP_MODELS" -eq 1 ]]; then
  log "Skipping models (--skip-models). Data ready in data/aggregations/"
  exit 0
fi

# --- steps 4–6: modeling ---
run_models ""

if [[ "$BY_CASE_TYPE" -eq 1 ]]; then
  run_models "cv"
  run_models "cr"
fi

log "Pipeline complete."
echo ""
echo "Outputs:"
echo "  data/aggregations/by_case.parquet (+ by_*_cv.parquet, by_*_cr.parquet with --by-case-type)"
echo "  docs/01_ml_results.json (+ _cv, _cr with --by-case-type)"
echo "  docs/02_xgb_shap_results.json (+ _cv, _cr)"
echo "  docs/03_causal_regression_results.json (+ _cv, _cr)"
echo "  reports/figures/"
