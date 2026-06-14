#!/usr/bin/env python3
"""
Full pipeline — federal court complexity project.

Research question: does dynamic judge workload at case opening predict
procedural complexity (complexity_index) beyond basic filing attributes?

Steps:
  1  build_features    scripts/build_features.py        (Event Log.csv → Event Log_model.csv)
  2  build_cases       src/build_case_features.py       (Event Log_model.csv → data/case_features.parquet)
  3  build_agg         src/build_aggregations.py        (case_features.parquet → data/aggregations/)
  4  rf                scripts/run_rf_shap.py           (RF: Model A vs B, cv + cr)
  5  xgb               scripts/run_xgb_shap.py          (XGBoost: Model A vs B, cv + cr)

Usage:
  .venv/bin/python3 scripts/run_pipeline.py
  .venv/bin/python3 scripts/run_pipeline.py --skip-clean --skip-case --skip-agg
  .venv/bin/python3 scripts/run_pipeline.py --skip-clean --skip-case --skip-agg --skip-rf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python3"


def _header(msg: str) -> None:
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def _run(label: str, cmd: list[str | Path]) -> None:
    _header(label)
    print(f"$ {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status}  ({elapsed:.1f}s)")
    if result.returncode != 0:
        sys.exit(f"\nPipeline aborted at: {label}")


def _check(path: Path, step: str) -> None:
    if not path.exists():
        sys.exit(f"\nMissing: {path.relative_to(ROOT)} — run without --skip-{step} first")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-clean",  action="store_true", help="Skip Step 1 (reuse Event Log_model.csv)")
    parser.add_argument("--skip-case",   action="store_true", help="Skip Step 2 (reuse case_features.parquet)")
    parser.add_argument("--skip-agg",    action="store_true", help="Skip Step 3 (reuse aggregations/)")
    parser.add_argument("--skip-rf",     action="store_true", help="Skip Step 4 (RF)")
    parser.add_argument("--skip-xgb",   action="store_true", help="Skip Step 5 (XGBoost)")
    parser.add_argument("--sample",      type=int, default=None, metavar="N",
                        help="Dev mode: limit event rows in steps 1-2")
    args = parser.parse_args()

    t_start = time.time()
    print(f"\nFederal Court Complexity Pipeline")
    print(f"Root: {ROOT}")

    # ── data steps ───────────────────────────────────────────────────────────
    if not args.skip_clean:
        _check(ROOT / "Event Log.csv", "clean")
        cmd = [PYTHON, "scripts/build_features.py"]
        if args.sample:
            cmd += ["--sample-rows", str(args.sample)]
        _run("Step 1 — build_features", cmd)
    else:
        _check(ROOT / "Event Log_model.csv", "case")
        print("[Step 1] Skipped")

    if not args.skip_case:
        _check(ROOT / "Event Log_model.csv", "case")
        cmd = [PYTHON, "src/build_case_features.py"]
        if args.sample:
            cmd += ["--sample-rows", str(args.sample)]
        _run("Step 2 — build_case_features", cmd)
    else:
        _check(ROOT / "data" / "case_features.parquet", "agg")
        print("[Step 2] Skipped")

    if not args.skip_agg:
        _check(ROOT / "data" / "case_features.parquet", "agg")
        _run("Step 3 — build_aggregations", [PYTHON, "src/build_aggregations.py"])
    else:
        _check(ROOT / "data" / "aggregations" / "by_case.parquet", "rf")
        print("[Step 3] Skipped")

    # ── models (cv and cr separately) ────────────────────────────────────────
    for case_type in ["cv", "cr"]:
        if not args.skip_rf:
            _run(f"Step 4 — RF [{case_type}]",
                 [PYTHON, "scripts/run_rf_shap.py", "--case-type", case_type])
        if not args.skip_xgb:
            _run(f"Step 5 — XGBoost [{case_type}]",
                 [PYTHON, "scripts/run_xgb_shap.py", "--case-type", case_type])

    _header("Pipeline complete")
    print(f"Total time: {time.time() - t_start:.1f}s")
    print("\nOutputs:")
    for p in sorted((ROOT / "docs").glob("0*.json")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
