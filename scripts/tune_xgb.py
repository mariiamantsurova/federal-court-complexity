#!/usr/bin/env python3
"""
Hyperparameter tuning for XGBoost via randomised search with TimeSeriesSplit.

Tunes on Model C (all features) of the cv case type (largest split).
Best params are saved to docs/xgb_best_params_{target}.json and reused by run_xgb.py.

Usage:

  .venv/bin/python3 scripts/tune_xgb.py --n-iter 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_dataset, prepare, TARGET

DOCS = ROOT / "docs"

PARAM_GRID = {
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":        [0.65, 0.75, 0.85, 0.95],
    "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
    "min_child_weight": [1, 3, 5, 10],
    "reg_alpha":        [0.0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

FIXED = dict(
    n_estimators=800,
    early_stopping_rounds=40,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


def _cv_mae(params: dict, X_train, y_train, n_splits: int = 3) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        X_tr,  X_val  = X_train.iloc[tr_idx],  X_train.iloc[val_idx]
        y_tr,  y_val  = y_train.iloc[tr_idx],  y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**{**FIXED, **params})
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        maes.append(mean_absolute_error(y_val, model.predict(X_val)))
    return float(np.mean(maes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=15,
                        help="Number of random parameter combinations to try")
    parser.add_argument("--case-type", default="cv",
                        help="Case type to tune on (default: cv — largest set)")
    args = parser.parse_args()

    df = load_dataset()
    # Tune on Model C (richest feature set) — best params generalise to A and B
    X_train, X_test, y_train, y_test = prepare(df, args.case_type, "C", target=TARGET)

    print(f"Tuning XGBoost on {args.case_type.upper()} Model C | target={TARGET} "
          f"({len(X_train):,} train rows, {X_train.shape[1]} features)")
    print(f"Running {args.n_iter} random combinations with 3-fold TimeSeriesSplit ...\n")

    rng = random.Random(42)
    combos = [
        {k: rng.choice(v) for k, v in PARAM_GRID.items()}
        for _ in range(args.n_iter)
    ]

    best_mae   = float("inf")
    best_params: dict = {}
    results    = []

    for i, params in enumerate(combos, 1):
        mae = _cv_mae(params, X_train, y_train)
        results.append({"params": params, "cv_mae": round(mae, 6)})
        marker = " ◀ best" if mae < best_mae else ""
        print(f"  [{i:2d}/{args.n_iter}]  MAE={mae:.5f}  {params}{marker}")
        if mae < best_mae:
            best_mae   = mae
            best_params = params

    print(f"\nBest CV MAE: {best_mae:.5f}")
    print(f"Best params: {best_params}")

    # Validate best params on the held-out test set
    model = xgb.XGBRegressor(**{**FIXED, **best_params})
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Test MAE with best params: {test_mae:.5f}")

    output = {
        "tuned_on":  args.case_type,
        "target":    TARGET,
        "best_cv_mae":   round(best_mae, 6),
        "best_test_mae": round(test_mae, 6),
        "best_params":   best_params,
        "fixed_params":  {k: v for k, v in FIXED.items()
                          if k not in ("early_stopping_rounds",)},
        "all_results":   sorted(results, key=lambda r: r["cv_mae"]),
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    out_path = DOCS / f"xgb_best_params_{TARGET}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
