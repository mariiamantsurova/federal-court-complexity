#!/usr/bin/env python3
"""
MDL sensitivity: re-run models excluding is_mdl == True, then rebuild summary tables.

Usage:
  python scripts/run_mdl_sensitivity.py
  python scripts/run_mdl_sensitivity.py --tables-only   # skip model runs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
FEATURES = ROOT / "data" / "aggregations" / "by_case.parquet"
CASE_TYPES = ("all", "cv", "cr")


def _run_model(case_type: str) -> None:
    ct_args = [] if case_type == "all" else ["--case-type", case_type]
    flag = ["--exclude-mdl"]
    cmds = [
        [str(PYTHON), "scripts/run_ml_importance.py", "--features", str(FEATURES), *ct_args, *flag],
        [str(PYTHON), "scripts/run_xgb_shap.py", "--features", str(FEATURES), *ct_args, *flag],
        [str(PYTHON), "scripts/run_causal_regression.py", "--features", str(FEATURES), *ct_args, *flag],
    ]
    for cmd in cmds:
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="Only rebuild docs/tables/ from existing JSON (skip model runs)",
    )
    args = p.parse_args()

    if not args.tables_only:
        if not FEATURES.is_file():
            raise FileNotFoundError(f"{FEATURES} missing — run the pipeline first.")
        print("MDL sensitivity: excluding is_mdl == True")
        for ct in CASE_TYPES:
            print(f"\n=== case_type={ct} ===")
            _run_model(ct)

    print("\n=== Building summary tables ===")
    subprocess.run([str(PYTHON), "scripts/build_results_tables.py"], cwd=ROOT, check=True)
    print(f"\nDone. Tables -> {ROOT / 'docs' / 'tables'}")


if __name__ == "__main__":
    main()
