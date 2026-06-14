#!/usr/bin/env python3
"""
XGBoost — Model A / B / C comparison for cv and cr case types.

Model A: filing attributes only  (no judge info)
Model B: + District_Judge_idx    (judge identity)
Model C: + judge workload cols   (judge_open_at_filing, judge_opened_30d, judge_closed_30d)

Loads tuned hyperparameters from docs/xgb_best_params.json if present
(run scripts/tune_xgb.py first to generate them).

Usage:
  .venv/bin/python3 scripts/run_xgb.py              # both case types
  .venv/bin/python3 scripts/run_xgb.py --case-type cv
  .venv/bin/python3 scripts/run_xgb.py --case-type cr

Outputs (docs/):
  xgb_results.json

Outputs (reports/figures/):
  xgb_mae_comparison.png          — A/B/C MAE grouped bar chart (cv + cr)
  xgb_shap_beeswarm_{ct}.png      — SHAP beeswarm, Model C
  xgb_shap_bar_{ct}_model{X}.png  — SHAP mean |value| bar, all models
  xgb_scatter_{ct}.png            — Actual vs Predicted, Model C
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
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_dataset, prepare, get_feature_sets

DOCS        = ROOT / "docs"
FIGURES     = ROOT / "reports" / "figures"
PARAMS_PATH = DOCS / "xgb_best_params.json"

DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.01,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


def _load_params() -> dict:
    if PARAMS_PATH.exists():
        data = json.loads(PARAMS_PATH.read_text())
        tuned = {**data["fixed_params"], **data["best_params"],
                 "random_state": 42, "n_jobs": -1, "tree_method": "hist"}
        print(f"Using tuned params from {PARAMS_PATH.name}  (cv_mae={data['best_cv_mae']:.5f})")
        return tuned
    print("No tuned params found — using defaults. Run scripts/tune_xgb.py first.")
    return DEFAULT_PARAMS


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_shap_bar(top15: list[dict], title: str, path: Path, color: str = "#2980b9"):
    features = [r["feature"] for r in top15][::-1]
    values   = [r["mean_abs_shap"] for r in top15][::-1]
    fig, ax  = plt.subplots(figsize=(9, 5))
    ax.barh(features, values, color=color)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_beeswarm(sv, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(y_true, y_pred, case_type: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=4, color="#2980b9", rasterized=True)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect")
    ax.set_xlabel("Actual event density (ed)")
    ax.set_ylabel("Predicted event density (ed)")
    ax.set_title(f"Actual vs Predicted — Model C | {case_type.upper()}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    ax.set_title("XGBoost: MAE by Model Level and Case Type")
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
    ax.set_title(f"R² by Model Level | {ct.upper()}")
    ax.set_ylim(0, min(1.0, max(r2s) * 1.12))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main logic ────────────────────────────────────────────────────────────────

def run_case_type(df, case_type: str, params: dict) -> dict:
    print(f"\n── XGBoost | {case_type.upper()} ──")
    models_out   = {}
    feature_sets = get_feature_sets(df)
    preds        = {}
    y_test_vals  = None

    for level in ["A", "B", "C"]:
        X_train, X_test, y_train, y_test = prepare(df, case_type, level)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        preds[level] = y_pred
        if y_test_vals is None:
            y_test_vals = np.asarray(y_test, dtype=np.float64)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))

        # SHAP
        sample_X  = X_test.sample(min(2000, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        sv        = explainer(sample_X)
        mean_abs  = np.abs(sv.values).mean(axis=0)
        top15     = sorted(
            zip(X_test.columns.tolist(), mean_abs.tolist()),
            key=lambda x: x[1], reverse=True
        )[:15]

        FIGURES.mkdir(parents=True, exist_ok=True)
        _plot_shap_bar(
            [{"feature": f, "mean_abs_shap": v} for f, v in top15],
            title=f"XGBoost Model {level} | {case_type.upper()}",
            path=FIGURES / f"xgb_shap_bar_{case_type}_model{level}.png",
        )
        if level == "C":
            _plot_beeswarm(sv, f"SHAP Beeswarm — Model C | {case_type.upper()}",
                           FIGURES / f"xgb_shap_beeswarm_{case_type}.png")

        models_out[level] = {
            "mae":        round(mae, 6),
            "r2":         round(r2, 6),
            "n_train":    int(len(X_train)),
            "n_test":     int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "features":   feature_sets[level],
            "shap_top15": [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in top15],
        }
        print(f"  Model {level}: MAE={mae:.4f}  R²={r2:.4f}  "
              f"(train={len(X_train):,}  test={len(X_test):,}  features={X_train.shape[1]})")

    # Persist predictions for downstream analysis
    DOCS.mkdir(parents=True, exist_ok=True)
    np.savez(DOCS / f"xgb_predictions_{case_type}.npz",
             y_test=y_test_vals, **{f"pred_{lv}": preds[lv] for lv in preds}) # type: ignore

    _plot_scatter(y_test_vals, preds["C"], case_type,
                  FIGURES / f"xgb_scatter_{case_type}.png")

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

    # R² waterfall
    _plot_r2_waterfall(result, FIGURES / f"xgb_r2_waterfall_{case_type}.png")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr", "both"], default="both")
    args = parser.parse_args()

    params = _load_params()
    df     = load_dataset()
    types  = ["cv", "cr"] if args.case_type == "both" else [args.case_type]

    all_results = {}
    for ct in types:
        all_results[ct] = run_case_type(df, ct, params)

    # Cross-case-type MAE comparison chart (only when both are run)
    if len(all_results) > 1:
        FIGURES.mkdir(parents=True, exist_ok=True)
        _plot_mae_comparison(all_results, FIGURES / "xgb_mae_comparison.png")
        print(f"\nSaved → {FIGURES / 'xgb_mae_comparison.png'}")

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "xgb_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
