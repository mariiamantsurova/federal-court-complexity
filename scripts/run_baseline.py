#!/usr/bin/env python3
"""
Judge-median baseline for LOS prediction.

For each test case, predict the judge's median LOS from training data.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(case_type: str | None, exclude_mdl: bool) -> pd.DataFrame:
    path = ROOT / "data" / "aggregations" / "by_case.parquet"
    df = pd.read_parquet(path)
    if case_type in ("cv", "cr"):
        df = df[df["case_type"] == case_type]
    if exclude_mdl and "is_mdl" in df.columns:
        df = df[~df["is_mdl"]]
    print(f"Loaded {len(df):,} cases  (case_type={case_type or 'all'}, exclude_mdl={exclude_mdl})")
    return df


def temporal_split(df: pd.DataFrame, cutoff_quantile: float = 0.8) -> tuple:
    dates = pd.to_datetime(df["case_open_date"])
    cutoff = dates.quantile(cutoff_quantile)
    return df[dates < cutoff].copy(), df[dates >= cutoff].copy(), cutoff.date()  # type: ignore


def build_suffix(case_type: str | None, exclude_mdl: bool) -> str:
    parts = []
    if case_type:
        parts.append(case_type)
    if exclude_mdl:
        parts.append("no_mdl")
    return ("_" + "_".join(parts)) if parts else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None)
    parser.add_argument("--exclude-mdl", action="store_true")
    args = parser.parse_args()

    suffix   = build_suffix(args.case_type, args.exclude_mdl)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    df = load_data(args.case_type, args.exclude_mdl)

    # Keep only closed cases with valid LOS
    df = df[df["los_days"].notna() & (df["los_days"] > 0)].copy()

    train_df, test_df, cutoff = temporal_split(df)
    print(f"Temporal split: cutoff={cutoff}  train={len(train_df):,}  test={len(test_df):,}")

    # ── Judge-median lookup from training set ─────────────────────────────────
    judge_medians = (
        train_df.groupby("District_Judge")["los_days"]
        .median()
        .to_dict()
    )
    overall_median = float(train_df["los_days"].median())
    print(f"Judges in training set: {len(judge_medians)}")
    print(f"Overall training median: {overall_median:.1f} days")

    # ── Predict ───────────────────────────────────────────────────────────────
    y_true = test_df["los_days"].values
    y_pred = np.array([
        judge_medians.get(judge, overall_median)
        for judge in test_df["District_Judge"]
    ])

    n_unseen = (test_df["District_Judge"].map(judge_medians).isna()).sum()
    print(f"Test cases using overall median (unseen judge): {n_unseen:,}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    print(f"\nBaseline metrics:  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

    # ── Overall-mean baseline for reference ──────────────────────────────────
    mean_pred = np.full_like(y_true, float(train_df["los_days"].mean()), dtype=float)
    mae_mean  = mean_absolute_error(y_true, mean_pred)
    r2_mean   = r2_score(y_true, mean_pred)
    print(f"Mean-only baseline:  MAE={mae_mean:.1f}  R²={r2_mean:.4f}  (R² should be 0.0)")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "model": "JudgeMedianBaseline",
        "case_type": args.case_type or "all",
        "exclude_mdl": args.exclude_mdl,
        "n_train": int(len(train_df)),
        "n_test":  int(len(test_df)),
        "n_judges_train": len(judge_medians),
        "n_test_unseen_judge": int(n_unseen),
        "overall_train_median": round(overall_median, 2),
        "metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)},
        "mean_only_baseline": {"MAE": round(mae_mean, 2), "R2": round(r2_mean, 4)},
    }
    json_path = docs_dir / f"00_baseline_results{suffix}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
