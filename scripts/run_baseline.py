#!/usr/bin/env python3
"""
Baseline for complexity_index prediction.

Predicts the judge's median complexity_index from training data.
Cases with an unseen judge fall back to the overall training median.
This is the minimum bar every model should beat.

Outputs:
  docs/00_baseline_results.json

Usage:
  .venv/bin/python3 scripts/run_baseline.py
  .venv/bin/python3 scripts/run_baseline.py --case-type cv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.features import TARGET, add_derived_columns


def load_data(case_type: str | None) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "aggregations" / "by_case.parquet")
    df = add_derived_columns(df)
    if case_type in ("cv", "cr"):
        df = df[df["case_type"] == case_type]
    df = df[df[TARGET].notna()].copy()
    print(f"Loaded {len(df):,} cases (case_type={case_type or 'all'})")
    return df


def temporal_split(df: pd.DataFrame, cutoff_quantile: float = 0.8) -> tuple:
    dates = pd.to_datetime(df["case_open_date"])
    cutoff = dates.quantile(cutoff_quantile)
    return df[dates < cutoff].copy(), df[dates >= cutoff].copy(), cutoff.date()  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None)
    args = parser.parse_args()

    suffix = f"_{args.case_type}" if args.case_type else ""
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    df = load_data(args.case_type)
    train_df, test_df, cutoff = temporal_split(df)
    print(f"Temporal split: cutoff={cutoff}  train={len(train_df):,}  test={len(test_df):,}")

    judge_medians = train_df.groupby("District_Judge")[TARGET].median().to_dict()
    overall_median = float(train_df[TARGET].median())
    print(f"Judges in training: {len(judge_medians)}  |  overall median: {overall_median:.3f}")

    y_true = test_df[TARGET].values
    y_pred = np.array([
        judge_medians.get(j, overall_median)
        for j in test_df["District_Judge"]
    ])

    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\nBaseline  MAE={mae:.4f}  R²={r2:.4f}")

    results = {
        "model": "JudgeMedianBaseline",
        "target": TARGET,
        "case_type": args.case_type or "all",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": {"MAE": round(mae, 4), "R2": round(r2, 4)},
    }
    out = docs_dir / f"00_baseline_results{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
