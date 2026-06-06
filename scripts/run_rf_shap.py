#!/usr/bin/env python3
"""
Random Forest with SHAP feature importance for LOS prediction.

Outputs (suffix = '' | '_cv' | '_cr' | '_no_mdl' | combinations):
  docs/01_ml_results{suffix}.json        — MAE, RMSE, R², top features
  docs/01_feature_importance{suffix}.csv — RF Gini importance per feature
  reports/figures/01_rf_feature_importance{suffix}.png

Usage:
  .venv/bin/python3 scripts/run_rf_shap.py
  .venv/bin/python3 scripts/run_rf_shap.py --case-type cv
  .venv/bin/python3 scripts/run_rf_shap.py --case-type cr --exclude-mdl
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None)
    parser.add_argument("--exclude-mdl", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--shap-sample", type=int, default=2000,
                        help="Number of test rows to use for SHAP (speed)")
    args = parser.parse_args()

    suffix = build_suffix(args.case_type, args.exclude_mdl)
    docs_dir = ROOT / "docs"
    figs_dir = ROOT / "reports" / "figures"
    docs_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    df = load_data(args.case_type, args.exclude_mdl)
    X, y = prepare_for_trees(df)
    print(f"Feature matrix: {X.shape[0]:,} × {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\nTraining RandomForest (n_estimators={args.n_estimators}) ...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    # ── metrics ───────────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)
    print(f"\nTest metrics:  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

    # ── feature importance (Gini) ─────────────────────────────────────────────
    importances = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": rf.feature_importances_,
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    imp_path = docs_dir / f"01_feature_importance{suffix}.csv"
    importances.to_csv(imp_path, index=False)
    print(f"Saved {imp_path.relative_to(ROOT)}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    print(f"\nComputing SHAP on {min(args.shap_sample, len(X_test)):,} test rows ...")
    t1 = time.time()
    shap_sample = X_test.iloc[: args.shap_sample]
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(shap_sample)
    print(f"  SHAP done in {time.time() - t1:.1f}s")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_n = 15
    fig, ax = plt.subplots(figsize=(8, 5))
    top = shap_df.head(top_n)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (days)")
    label = {"cv": "Civil", "cr": "Criminal"}.get(args.case_type, "All")
    mdl_tag = " (excl. MDL)" if args.exclude_mdl else ""
    ax.set_title(f"RF SHAP – Top {top_n} Features  [{label}{mdl_tag}]")
    fig.tight_layout()
    fig_path = figs_dir / f"01_rf_feature_importance{suffix}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Saved {fig_path.relative_to(ROOT)}")

    # ── results JSON ──────────────────────────────────────────────────────────
    results = {
        "model": "RandomForest",
        "case_type": args.case_type or "all",
        "exclude_mdl": args.exclude_mdl,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "max_depth": 15,
            "min_samples_leaf": 5,
        },
        "metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)},
        "top_features_gini": importances["feature"].head(10).tolist(),
        "top_features_shap": shap_df["feature"].head(10).tolist(),
    }
    json_path = docs_dir / f"01_ml_results{suffix}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {json_path.relative_to(ROOT)}")

    print(f"\nDone.  Top SHAP features: {shap_df['feature'].head(5).tolist()}")


if __name__ == "__main__":
    main()
