#!/usr/bin/env python3
"""
XGBoost with SHAP explanations for LOS prediction.

Outputs:
  docs/02_xgb_shap_results{suffix}.json   — MAE, RMSE, R², top features
  docs/02_xgb_shap_importance{suffix}.csv — mean |SHAP| per feature
  reports/figures/02_xgb_shap_bar{suffix}.png
  reports/figures/02_xgb_shap_beeswarm{suffix}.png

Usage:
  .venv/bin/python3 scripts/run_xgb_shap.py
  .venv/bin/python3 scripts/run_xgb_shap.py --case-type cv
  .venv/bin/python3 scripts/run_xgb_shap.py --case-type cr --exclude-mdl
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing_trees import prepare_for_trees


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(case_type: str | None, exclude_mdl: bool) -> pd.DataFrame:
    path = ROOT / "data" / "aggregations" / "by_case.parquet"
    df = pd.read_parquet(path)
    if case_type in ("cv", "cr"):
        df = df[df["case_type"] == case_type]
    if exclude_mdl and "is_mdl" in df.columns:
        df = df[~df["is_mdl"]]
    print(f"Loaded {len(df):,} cases  (case_type={case_type or 'all'}, exclude_mdl={exclude_mdl})")
    return df


def build_suffix(case_type: str | None, exclude_mdl: bool) -> str:
    parts = []
    if case_type:
        parts.append(case_type)
    if exclude_mdl:
        parts.append("no_mdl")
    return ("_" + "_".join(parts)) if parts else ""


def temporal_split(df: pd.DataFrame, cutoff_quantile: float = 0.8) -> tuple:
    """Split by case_open_date so test cases are always newer than training cases."""
    dates = pd.to_datetime(df["case_open_date"])
    cutoff = dates.quantile(cutoff_quantile)
    return df[dates < cutoff].copy(), df[dates >= cutoff].copy(), cutoff.date()


def _label(case_type: str | None, exclude_mdl: bool) -> str:
    name = {"cv": "Civil", "cr": "Criminal"}.get(case_type or "", "All")
    return name + (" (excl. MDL)" if exclude_mdl else "")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None)
    parser.add_argument("--exclude-mdl", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--shap-sample", type=int, default=2000,
                        help="Rows used for SHAP plots (beeswarm needs ≤ few k)")
    args = parser.parse_args()

    suffix = build_suffix(args.case_type, args.exclude_mdl)
    label  = _label(args.case_type, args.exclude_mdl)
    docs_dir = ROOT / "docs"
    figs_dir = ROOT / "reports" / "figures"
    docs_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    df = load_data(args.case_type, args.exclude_mdl)
    train_df, test_df, cutoff = temporal_split(df)
    print(f"Temporal split: cutoff={cutoff}  train={len(train_df):,}  test={len(test_df):,}")

    # Build judge target encoding from training data only, then apply to both splits.
    judge_target_map = (
        train_df.dropna(subset=["log_los_days"])
        .groupby("District_Judge")["log_los_days"]
        .median()
        .to_dict()
    )
    print(f"Judge target encoding: {len(judge_target_map)} judges in training set")

    X_train, y_train = prepare_for_trees(train_df, target="log_los_days", judge_target_map=judge_target_map)
    X_test,  y_test  = prepare_for_trees(test_df,  target="log_los_days", judge_target_map=judge_target_map)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print(f"Feature matrix: {X_train.shape[1]} features")

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\nTraining XGBoost (n_estimators={args.n_estimators}) ...")
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"  Training done in {time.time() - t0:.1f}s")

    # ── metrics ───────────────────────────────────────────────────────────────
    y_pred_log  = model.predict(X_test)
    y_pred_days = np.exp(y_pred_log)
    y_true_days = np.exp(y_test.values)
    mae  = mean_absolute_error(y_true_days, y_pred_days)
    rmse = mean_squared_error(y_true_days, y_pred_days) ** 0.5
    r2   = r2_score(y_true_days, y_pred_days)
    print(f"\nTest metrics:  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    n_shap = min(args.shap_sample, len(X_test))
    print(f"\nComputing SHAP on {n_shap:,} test rows ...")
    t1 = time.time()
    shap_sample = X_test.iloc[:n_shap]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    # SHAP values are in log-days (model predicts log_los_days)
    print(f"  SHAP done in {time.time() - t1:.1f}s")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    imp_path = docs_dir / f"02_xgb_shap_importance{suffix}.csv"
    shap_df.to_csv(imp_path, index=False)
    print(f"Saved {imp_path.relative_to(ROOT)}")

    # bar plot
    top_n = 15
    fig, ax = plt.subplots(figsize=(8, 5))
    top = shap_df.head(top_n)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#DD8452")
    ax.set_xlabel("Mean |SHAP value| (log-days)")
    ax.set_title(f"XGBoost SHAP – Top {top_n} Features  [{label}]")
    fig.tight_layout()
    bar_path = figs_dir / f"02_xgb_shap_bar{suffix}.png"
    fig.savefig(bar_path, dpi=120)
    plt.close(fig)
    print(f"Saved {bar_path.relative_to(ROOT)}")

    # beeswarm
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        shap_sample,
        max_display=15,
        show=False,
        plot_type="dot",
    )
    plt.title(f"XGBoost SHAP Beeswarm  [{label}]")
    plt.tight_layout()
    bee_path = figs_dir / f"02_xgb_shap_beeswarm{suffix}.png"
    plt.savefig(bee_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {bee_path.relative_to(ROOT)}")

    # ── results JSON ──────────────────────────────────────────────────────────
    results = {
        "model": "XGBoost",
        "case_type": args.case_type or "all",
        "exclude_mdl": args.exclude_mdl,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        "metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)},
        "top_features_shap": shap_df["feature"].head(10).tolist(),
        "mean_abs_shap": {
            row["feature"]: round(row["mean_abs_shap"], 4)
            for _, row in shap_df.head(10).iterrows()
        },
    }
    json_path = docs_dir / f"02_xgb_shap_results{suffix}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {json_path.relative_to(ROOT)}")

    print(f"\nDone.  Top SHAP features: {shap_df['feature'].head(5).tolist()}")


if __name__ == "__main__":
    main()
