#!/usr/bin/env python3
"""
Ridge Regression — Model A / B / C comparison for cv and cr case types.

Model A: filing attributes only
Model B: + District_Judge  (judge identity)
Model C: + judge_open_at_filing, judge_opened_30d, judge_closed_30d  (workload)

Ridge is a linear model, so each pipeline is:
  median impute  →  standardize  →  Ridge(alpha)

The hyperparameter is the L2 penalty `alpha`. By default it is tuned per
(case_type × model level) with RidgeCV over a log-spaced grid; pass --alpha to
fix it instead. Feature importance is the standardized coefficient (features are
scaled, so coefficients are directly comparable in magnitude).

Usage:
  .venv/bin/python3 scripts/run_ridge_regression.py                       # both case types
  .venv/bin/python3 scripts/run_ridge_regression.py --case-type cv
  .venv/bin/python3 scripts/run_ridge_regression.py --case-type cr --alpha 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    load_dataset, prepare, get_feature_sets, TARGET,
)

DOCS    = ROOT / "docs"
FIGURES = ROOT / "reports" / "figures"

# Log-spaced alpha grid for RidgeCV when --alpha is not supplied.
ALPHA_GRID = np.logspace(-3, 4, 40)

CATEGORICAL_COLS = ["District_Judge"]


def _plot_mae_comparison(all_results: dict, path: Path):
    """Grouped bar chart of MAE for A/B/C across case types."""
    case_types = list(all_results.keys())
    levels     = ["A", "B", "C"]
    x          = np.arange(len(levels))
    width      = 0.35
    colors     = ["#2980b9", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ct, color) in enumerate(zip(case_types, colors)):
        maes = [all_results[ct]["models"][lv]["mae"] for lv in levels]
        bars = ax.bar(x + i * width, maes, width, label=ct.upper(), color=color, alpha=0.85)
        for bar, v in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Model A\n(filing only)", "Model B\n(+ judge ID)", "Model C\n(+ workload)"])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Ridge Regression: MAE by Model Level and Case Type")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_r2_waterfall(result: dict, path: Path):
    """R² improvement waterfall A → B → C for one case type."""
    ct     = result["case_type"]
    levels = ["A", "B", "C"]
    r2s    = [result["models"][lv]["r2"] for lv in levels]
    labels = ["Model A\n(filing only)", "Model B\n(+ judge ID)", "Model C\n(+ workload)"]
    colors = ["#2980b9", "#27ae60" if r2s[1] >= r2s[0] else "#e74c3c",
              "#27ae60" if r2s[2] >= r2s[1] else "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, r2s, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"R²={v:.4f}", ha="center", va="bottom", fontsize=9)
    for i in range(1, len(r2s)):
        delta = r2s[i] - r2s[i - 1]
        sign  = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.4f}", xy=(i, r2s[i]), xytext=(i - 0.5, max(r2s) * 1.02),
                    fontsize=8, color="grey", ha="center")

    ax.set_ylabel("R²")
    ax.set_title(f"Ridge Regression: R² by Model Level | {ct.upper()}")
    ax.set_ylim(min(0, min(r2s)), min(1.0, max(r2s) * 1.12) if max(r2s) > 0 else 0.05)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case_type(df, case_type: str, target: str, alpha: float | None) -> dict:
    print(f"\n── Ridge Regression | {case_type.upper()} | target={target} ──")
    models_out = {}

    feature_sets = get_feature_sets(df)

    for level in ["A", "B", "C"]:
        X_train, X_test, y_train, y_test = prepare(df, case_type, level, target=target)

        if alpha is None:
            regressor = RidgeCV(alphas=ALPHA_GRID)
        else:
            regressor = Ridge(alpha=alpha)

        # Numeric features: median-impute + standardize. Nominal id columns
        # (District_Judge) are one-hot encoded instead — handle_unknown
        # ignores judges that appear only in the newer test split.
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])

        model = Pipeline([
            ("prep",  preprocessor),
            ("ridge", regressor),
        ])
        model.fit(X_train, y_train)

        ridge = model.named_steps["ridge"]
        chosen_alpha = float(getattr(ridge, "alpha_", getattr(ridge, "alpha", np.nan)))

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))

        # Coefficients → feature importance. Numeric features are standardized so
        # their coefficients are comparable; one-hot judge dummies each get their
        # own coefficient. Strip the ColumnTransformer's num__/cat__ prefixes.
        feat_names = [n.split("__", 1)[-1]
                      for n in model.named_steps["prep"].get_feature_names_out()]
        coefs = ridge.coef_
        top15 = sorted(
            zip(feat_names, coefs.tolist()),
            key=lambda x: abs(x[1]), reverse=True
        )[:15]

        models_out[level] = {
            "mae":      round(mae, 6),
            "r2":       round(r2, 6),
            "alpha":    round(chosen_alpha, 6),
            "n_train":  int(len(X_train)),
            "n_test":   int(len(X_test)),
            "features": feature_sets[level],
            "coef_top15": [
                {"feature": f, "coef": round(v, 6)} for f, v in top15
            ],
        }
        print(f"  Model {level}: MAE={mae:.4f}  R²={r2:.4f}  alpha={chosen_alpha:.4g}  "
              f"(train={len(X_train):,}  test={len(X_test):,})")

    m = models_out
    result = {
        "case_type": case_type,
        "target":    target,
        "models":    m,
        "B_vs_A_mae_improvement": round(m["A"]["mae"] - m["B"]["mae"], 6),
        "C_vs_B_mae_improvement": round(m["B"]["mae"] - m["C"]["mae"], 6),
        "C_vs_A_mae_improvement": round(m["A"]["mae"] - m["C"]["mae"], 6),
    }
    print(f"  ΔB-A={result['B_vs_A_mae_improvement']:+.4f}  "
          f"ΔC-B={result['C_vs_B_mae_improvement']:+.4f}  "
          f"ΔC-A={result['C_vs_A_mae_improvement']:+.4f}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_r2_waterfall(result, FIGURES / f"ridge_r2_waterfall_{target}_{case_type}.png")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Fixed L2 penalty. Omit to tune per model with RidgeCV.")
    args = parser.parse_args()

    df = load_dataset()
    types = ["cv", "cr"] if args.case_type == "both" else [args.case_type]

    all_results = {}
    for ct in types:
        all_results[ct] = run_case_type(df, ct, TARGET, args.alpha)

    # Cross-case-type MAE comparison chart (only when both are run)
    if len(all_results) > 1:
        FIGURES.mkdir(parents=True, exist_ok=True)
        _plot_mae_comparison(all_results, FIGURES / f"ridge_mae_comparison_{TARGET}.png")
        print(f"\nSaved → {FIGURES / f'ridge_mae_comparison_{TARGET}.png'}")

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"ridge_results_{TARGET}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
