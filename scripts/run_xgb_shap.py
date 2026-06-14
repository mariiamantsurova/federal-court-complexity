#!/usr/bin/env python3
"""
XGBoost — Model A vs Model B comparison for complexity_index prediction.

  Model A: case filing features only
  Model B: filing features + judge_workload_at_open

Run separately for civil (cv) and criminal (cr) as recommended.

Outputs:
  docs/02_xgb_results{suffix}.json
  reports/figures/02_xgb_shap_{a|b}{suffix}.png

Usage:
  .venv/bin/python3 scripts/run_xgb_shap.py --case-type cv
  .venv/bin/python3 scripts/run_xgb_shap.py --case-type cr
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from src.features import TARGET, add_derived_columns
from src.judge_workload import add_judge_workload
from src.preprocessing_trees import prepare_for_trees


def load_data(case_type: str | None) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "aggregations" / "by_case.parquet")
    df = add_derived_columns(df)
    if case_type in ("cv", "cr"):
        df = df[df["case_type"] == case_type]
    print(f"Loaded {len(df):,} cases (case_type={case_type or 'all'})")
    return df


def temporal_split(df: pd.DataFrame, cutoff_quantile: float = 0.8) -> tuple:
    dates = pd.to_datetime(df["case_open_date"])
    cutoff = dates.quantile(cutoff_quantile)
    return df[dates < cutoff].copy(), df[dates >= cutoff].copy(), cutoff.date()  # type: ignore


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    include_workload: bool,
    n_estimators: int,
    label: str,
    figs_dir: Path,
    shap_sample: int,
) -> dict:
    X_train, y_train = prepare_for_trees(train_df, include_workload=include_workload)
    X_test,  y_test  = prepare_for_trees(test_df,  include_workload=include_workload)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\n[{label}] Training XGBoost ({X_train.shape[1]} features) ...")
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  done in {time.time() - t0:.1f}s")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"  MAE={mae:.4f}  R²={r2:.4f}")

    n_shap = min(shap_sample, len(X_test))
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:n_shap])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # bar plot
    top_n = 15
    fig, ax = plt.subplots(figsize=(8, 5))
    top = shap_df.head(top_n)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#DD8452")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"XGBoost SHAP — {label} — Top {top_n} Features")
    fig.tight_layout()
    tag = label.lower().replace(" ", "_").replace("+", "plus")
    fig_path = figs_dir / f"02_xgb_shap_{tag}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    # beeswarm
    bee_path = figs_dir / f"02_xgb_beeswarm_{tag}.png"
    shap.summary_plot(shap_values, X_test.iloc[:n_shap], max_display=15, show=False, plot_type="dot")
    plt.title(f"XGBoost SHAP Beeswarm — {label}")
    plt.tight_layout()
    plt.savefig(bee_path, dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_features": int(X_train.shape[1]),
        "top_features_shap": shap_df["feature"].head(10).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--shap-sample", type=int, default=2000)
    args = parser.parse_args()

    suffix = f"_{args.case_type}" if args.case_type else ""
    docs_dir = ROOT / "docs"
    figs_dir = ROOT / "reports" / "figures"
    docs_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.case_type)
    print("Adding judge workload feature ...")
    df = add_judge_workload(df)
    train_df, test_df, cutoff = temporal_split(df)
    print(f"Temporal split: cutoff={cutoff}  train={len(train_df):,}  test={len(test_df):,}")

    shared = dict(
        n_estimators=args.n_estimators,
        figs_dir=figs_dir,
        shap_sample=args.shap_sample,
    )

    result_a = train_and_eval(train_df, test_df, include_workload=False,
                              label="Model A (no workload)", **shared)
    result_b = train_and_eval(train_df, test_df, include_workload=True,
                              label="Model B (+workload)", **shared)

    improvement = round(result_a["mae"] - result_b["mae"], 4)
    print(f"\nMAE improvement from workload: {improvement:+.4f}")

    results = {
        "model": "XGBoost",
        "target": TARGET,
        "case_type": args.case_type or "all",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "hyperparams": {"n_estimators": args.n_estimators, "max_depth": 6, "learning_rate": 0.05},
        "model_A": result_a,
        "model_B": result_b,
        "workload_mae_improvement": improvement,
    }
    out = docs_dir / f"02_xgb_results{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
