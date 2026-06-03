#!/usr/bin/env python3
"""
Train Decision Tree and Random Forest regressors to predict LOS from complexity features.

Usage:
  python scripts/run_ml_importance.py
  python scripts/run_ml_importance.py --case-type cv
  python scripts/run_ml_importance.py --features data/aggregations/by_case_cr.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    TARGET,
    VALID_CASE_TYPES,
    apply_data_filters,
    make_preprocessor,
    prepare_case_model_frame,
    tagged_path,
)

DEFAULT_FEATURES = ROOT / "data" / "aggregations" / "by_case.parquet"
FALLBACK_FEATURES = ROOT / "data" / "case_features.parquet"
FIG_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            out = trans.get_feature_names_out(cols)
            names.extend(out.tolist())
        else:
            names.extend(cols if isinstance(cols, list) else [cols])
    return names


def _importance_frame(model, feature_names: list[str]) -> pd.DataFrame:
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _plot_importance(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 25) -> None:
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    sns.barplot(data=top, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _resolve_features_path(features_path: Path, case_type: str | None) -> Path:
    if case_type in VALID_CASE_TYPES:
        tagged = features_path.parent / f"by_case_{case_type}.parquet"
        if tagged.is_file():
            return tagged
    return features_path


def run_ml_importance(
    features_path: Path,
    *,
    case_type: str | None = None,
    exclude_mdl: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 12,
    rf_estimators: int = 200,
) -> dict:
    features_path = _resolve_features_path(features_path, case_type)
    if not features_path.is_file():
        features_path = FALLBACK_FEATURES
    if not features_path.is_file():
        raise FileNotFoundError("Run src/build_aggregations.py first.")

    raw = apply_data_filters(
        pd.read_parquet(features_path),
        case_type=case_type,
        exclude_mdl=exclude_mdl,
    )
    X, y, numeric, categorical = prepare_case_model_frame(raw)

    if "case_open_date" in raw.columns:
        dates = pd.to_datetime(raw.loc[X.index, "case_open_date"], errors="coerce")
        order = dates.sort_values().index
        split_at = int(len(order) * (1 - test_size))
        train_idx, test_idx = order[:split_at], order[split_at:]
        split_type = "time (case_open_date)"
    else:
        train_idx, test_idx = train_test_split(
            X.index, test_size=test_size, random_state=random_state
        )
        split_type = "random"

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    preprocessor = make_preprocessor(numeric, categorical)

    models = {
        "decision_tree": DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=50,
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=max_depth,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    label = case_type or "all"
    mdl_label = "no_mdl" if exclude_mdl else "all_cases"
    results: dict = {
        "target": TARGET,
        "case_type": label,
        "exclude_mdl": exclude_mdl,
        "sample": mdl_label,
        "n_cases": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split": split_type,
        "models": {},
    }

    importance_tables: dict[str, pd.DataFrame] = {}

    for name, reg in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        prep = pipe.named_steps["prep"]
        feat_names = _feature_names(prep)
        imp_df = _importance_frame(pipe.named_steps["model"], feat_names)
        importance_tables[name] = imp_df

        results["models"][name] = {
            "metrics": _metrics(y_test.to_numpy(), pred),
            "top_features": imp_df.head(15).to_dict(orient="records"),
        }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    title_suffix = f" [{label}]" if label != "all" else ""
    if exclude_mdl:
        title_suffix += " (excl. MDL)"
    _plot_importance(
        importance_tables["decision_tree"],
        f"Decision Tree — feature importance (complexity → LOS){title_suffix}",
        tagged_path(FIG_DIR / "01_dt_feature_importance.png", case_type, exclude_mdl=exclude_mdl),
    )
    _plot_importance(
        importance_tables["random_forest"],
        f"Random Forest — feature importance (complexity → LOS){title_suffix}",
        tagged_path(FIG_DIR / "01_rf_feature_importance.png", case_type, exclude_mdl=exclude_mdl),
    )

    imp_path = tagged_path(DOCS_DIR / "01_feature_importance.csv", case_type, exclude_mdl=exclude_mdl)
    combined = importance_tables["random_forest"].copy()
    combined = combined.rename(columns={"importance": "rf_importance"})
    combined["dt_importance"] = importance_tables["decision_tree"].set_index("feature").reindex(
        combined["feature"]
    )["importance"].to_numpy()
    combined.to_csv(imp_path, index=False)

    results_path = tagged_path(DOCS_DIR / "01_ml_results.json", case_type, exclude_mdl=exclude_mdl)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"case_type={label} | cases: {len(X):,} | split: {split_type}")
    for name, info in results["models"].items():
        m = info["metrics"]
        print(f"\n{name}:")
        print(f"  MAE={m['mae']:.1f} days  RMSE={m['rmse']:.1f}  R²={m['r2']:.3f}")

    print(f"\nSaved -> {results_path}")
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
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--rf-estimators", type=int, default=200)
    args = p.parse_args()

    ct = None if args.case_type == "all" else args.case_type
    run_ml_importance(
        args.features,
        case_type=ct,
        exclude_mdl=args.exclude_mdl,
        test_size=args.test_size,
        random_state=args.random_state,
        max_depth=args.max_depth,
        rf_estimators=args.rf_estimators,
    )


if __name__ == "__main__":
    main()
