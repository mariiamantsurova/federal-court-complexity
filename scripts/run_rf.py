#!/usr/bin/env python3
"""
Random Forest — Model A / B / C comparison for cv and cr case types.

Model A: filing attributes only
Model B: + District_Judge_idx  (judge identity)
Model C: + judge_open_at_filing, judge_opened_30d, judge_closed_30d  (workload)

Usage:
  .venv/bin/python3 scripts/run_rf.py              # both case types
  .venv/bin/python3 scripts/run_rf.py --case-type cv
  .venv/bin/python3 scripts/run_rf.py --case-type cr
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_dataset, prepare, get_feature_sets

DOCS    = ROOT / "docs"
FIGURES = ROOT / "reports" / "figures"

RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)


def _shap_bar(features, values, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(features[::-1], values[::-1], color="#27ae60")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case_type(df, case_type: str) -> dict:
    print(f"\n── Random Forest | {case_type.upper()} ──")
    models_out = {}

    feature_sets = get_feature_sets(df)

    for level in ["A", "B", "C"]:
        X_train, X_test, y_train, y_test = prepare(df, case_type, level)

        # RF does not handle NaN natively — impute with column median
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp  = imputer.transform(X_test)

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train_imp, y_train)

        y_pred = model.predict(X_test_imp)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))

        # SHAP (sample to keep runtime reasonable)
        n_sample = min(500, len(X_test))
        sample_idx = np.random.default_rng(42).choice(len(X_test), n_sample, replace=False)
        sample_X = X_test_imp[sample_idx] # type: ignore
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample_X)
        mean_abs = np.abs(sv).mean(axis=0)
        top15 = sorted(
            zip(X_test.columns.tolist(), mean_abs.tolist()),
            key=lambda x: x[1], reverse=True
        )[:15]

        FIGURES.mkdir(parents=True, exist_ok=True)
        _shap_bar(
            [f for f, _ in top15], [v for _, v in top15],
            title=f"Random Forest Model {level} | {case_type.upper()} — Top 15 features",
            path=FIGURES / f"rf_shap_{case_type}_model{level}.png",
        )

        models_out[level] = {
            "mae":      round(mae, 6),
            "r2":       round(r2, 6),
            "n_train":  int(len(X_train)),
            "n_test":   int(len(X_test)),
            "features": feature_sets[level],
            "shap_top15": [
                {"feature": f, "mean_abs_shap": round(v, 6)} for f, v in top15
            ],
        }
        print(f"  Model {level}: MAE={mae:.4f}  R²={r2:.4f}  "
              f"(train={len(X_train):,}  test={len(X_test):,})")

    m = models_out
    result = {
        "case_type": case_type,
        "models":    m,
        "B_vs_A_mae_improvement": round(m["A"]["mae"] - m["B"]["mae"], 6),
        "C_vs_B_mae_improvement": round(m["B"]["mae"] - m["C"]["mae"], 6),
        "C_vs_A_mae_improvement": round(m["A"]["mae"] - m["C"]["mae"], 6),
    }
    print(f"  ΔB-A={result['B_vs_A_mae_improvement']:+.4f}  "
          f"ΔC-B={result['C_vs_B_mae_improvement']:+.4f}  "
          f"ΔC-A={result['C_vs_A_mae_improvement']:+.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr", "both"], default="both")
    args = parser.parse_args()

    df = load_dataset()
    types = ["cv", "cr"] if args.case_type == "both" else [args.case_type]

    all_results = {}
    for ct in types:
        all_results[ct] = run_case_type(df, ct)

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "rf_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
