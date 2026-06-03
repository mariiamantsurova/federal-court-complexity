#!/usr/bin/env python3
"""
XGBoost regression on case-level complexity → LOS with SHAP explanations.

Usage:
  python scripts/run_xgb_shap.py
  python scripts/run_xgb_shap.py --case-type cv
  python scripts/run_xgb_shap.py --shap-sample 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    TARGET,
    VALID_CASE_TYPES,
    apply_data_filters,
    feature_names_from_preprocessor,
    make_preprocessor,
    prepare_case_model_frame,
    tagged_path,
)

DEFAULT_FEATURES = ROOT / "data" / "aggregations" / "by_case.parquet"
FALLBACK_FEATURES = ROOT / "data" / "case_features.parquet"
FIG_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"


def _resolve_features_path(features_path: Path, case_type: str | None) -> Path:
    if case_type in VALID_CASE_TYPES:
        tagged = features_path.parent / f"by_case_{case_type}.parquet"
        if tagged.is_file():
            return tagged
    return features_path


def _time_split(raw: pd.DataFrame, X: pd.DataFrame, test_size: float):
    if "case_open_date" not in raw.columns:
        idx = X.index.to_numpy()
        tr, te = train_test_split(idx, test_size=test_size, random_state=42)
        return tr, te, "random"
    dates = pd.to_datetime(raw.loc[X.index, "case_open_date"], errors="coerce")
    order = dates.sort_values().index
    split_at = int(len(order) * (1 - test_size))
    return order[:split_at], order[split_at:], "time (case_open_date)"


def run_xgb_shap(
    features_path: Path,
    *,
    case_type: str | None = None,
    exclude_mdl: bool = False,
    test_size: float = 0.2,
    shap_sample: int = 3000,
    random_state: int = 42,
) -> dict:
    features_path = _resolve_features_path(features_path, case_type)
    if not features_path.is_file():
        features_path = FALLBACK_FEATURES
    if not features_path.is_file():
        raise FileNotFoundError("Run src/build_aggregations.py or src/build_case_features.py first.")

    raw = apply_data_filters(
        pd.read_parquet(features_path),
        case_type=case_type,
        exclude_mdl=exclude_mdl,
    )
    X, y, numeric, categorical = prepare_case_model_frame(raw)
    train_idx, test_idx, split_type = _time_split(raw, X, test_size)

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    preprocessor = make_preprocessor(numeric, categorical)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feat_names = feature_names_from_preprocessor(preprocessor)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train_t, y_train, eval_set=[(X_test_t, y_test)], verbose=False)

    pred = model.predict(X_test_t)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }

    n_shap = min(shap_sample, len(X_test_t))
    rng = np.random.default_rng(random_state)
    shap_idx = rng.choice(len(X_test_t), size=n_shap, replace=False)
    X_shap = X_test_t[shap_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    label = case_type or "all"
    title_suffix = f" [{label}]" if label != "all" else ""
    if exclude_mdl:
        title_suffix += " (excl. MDL)"

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feat_names, show=False, max_display=25)
    fig = plt.gcf()
    fig.tight_layout()
    beeswarm_path = tagged_path(FIG_DIR / "02_xgb_shap_beeswarm.png", case_type, exclude_mdl=exclude_mdl)
    fig.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_csv = tagged_path(DOCS_DIR / "02_xgb_shap_importance.csv", case_type, exclude_mdl=exclude_mdl)
    shap_df.to_csv(shap_csv, index=False)

    top = shap_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="darkorange")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"XGBoost — SHAP feature importance (complexity → LOS){title_suffix}")
    fig.tight_layout()
    bar_path = tagged_path(FIG_DIR / "02_xgb_shap_bar.png", case_type, exclude_mdl=exclude_mdl)
    fig.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    results = {
        "model": "xgboost",
        "target": TARGET,
        "case_type": label,
        "exclude_mdl": exclude_mdl,
        "sample": "no_mdl" if exclude_mdl else "all_cases",
        "n_cases": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "shap_sample": int(n_shap),
        "split": split_type,
        "metrics": metrics,
        "top_shap_features": shap_df.head(15).to_dict(orient="records"),
    }
    results_path = tagged_path(DOCS_DIR / "02_xgb_shap_results.json", case_type, exclude_mdl=exclude_mdl)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"case_type={label} | cases: {len(X):,} | split: {split_type} | SHAP n={n_shap:,}")
    print(f"XGBoost: MAE={metrics['mae']:.1f}d  RMSE={metrics['rmse']:.1f}  R²={metrics['r2']:.3f}")
    print(f"Saved -> {results_path}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument(
        "--case-type",
        choices=("all", *VALID_CASE_TYPES),
        default="all",
        help="Model subset: all (pooled), cv (civil), or cr (criminal)",
    )
    p.add_argument(
        "--exclude-mdl",
        action="store_true",
        help="Exclude Multi-District Litigation cases (is_mdl == True)",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--shap-sample", type=int, default=3000)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    ct = None if args.case_type == "all" else args.case_type
    run_xgb_shap(
        args.features,
        case_type=ct,
        exclude_mdl=args.exclude_mdl,
        test_size=args.test_size,
        shap_sample=args.shap_sample,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
