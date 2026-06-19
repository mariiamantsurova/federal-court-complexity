#!/usr/bin/env python3
"""
XGBoost — Model 1 / 2 / 3 / 4 comparison for cv and cr case types.

Model 1: filing attributes only  (no judge info)
Model 2: + judge workload cols    (workload, no judge identity)
Model 3: + District_Judge         (judge identity, no workload)
Model 4: + District_Judge + judge workload cols

Workload cols: open_cases_at_filing, aged_open_cases_at_filing,
clearance_rate_last_180_days. Models 2 and 4 isolate the workload signal with and
without judge identity.

Loads tuned hyperparameters from docs/xgb_best_params.json if present
(run scripts/tune_xgb.py first to generate them).

Usage:
  .venv/bin/python3 scripts/run_xgb.py              # both case types
  .venv/bin/python3 scripts/run_xgb.py --case-type cv
  .venv/bin/python3 scripts/run_xgb.py --case-type cr

Outputs (docs/):
  xgb_results.json

Outputs (reports/figures/):
  xgb_mae_comparison.png          — Model 1-4 MAE grouped bar chart (cv + cr)
  xgb_shap_beeswarm_{ct}.png      — SHAP beeswarm, Model 4
  xgb_shap_charts_{ct}.png        — SHAP importance bar charts, Models 1-4 in one figure
  xgb_scatter_{ct}.png            — Actual vs Predicted, Model 4
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

from src.preprocessing import (
    load_dataset, prepare, get_feature_sets, TARGET,
)

DOCS        = ROOT / "docs"
FIGURES     = ROOT / "reports" / "figures"

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
    enable_categorical=True,
)

CATEGORICAL_COLS = ["District_Judge"]

LEVELS       = ["1", "2", "3", "4"]
LEVEL_LABELS = ["Model 1\n(case only)", "Model 2\n(+ workload)",
                "Model 3\n(+ judge ID)", "Model 4\n(+ both)"]
RICHEST      = "4"   # full model used for SHAP / scatter


def _as_categorical(df):
    """Cast categorical id columns on the full frame so train/test share codes."""
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _load_params(target: str) -> dict:
    params_path = DOCS / f"xgb_best_params_{target}.json"
    if params_path.exists():
        data = json.loads(params_path.read_text())
        tuned = {**data["fixed_params"], **data["best_params"],
                 "random_state": 42, "n_jobs": -1, "tree_method": "hist",
                 "enable_categorical": True}
        print(f"Using tuned params from {params_path.name}  (cv_mae={data['best_cv_mae']:.5f})")
        return tuned
    print(f"No tuned params for target='{target}' — using defaults. "
          f"Run scripts/tune_xgb.py first.")
    return DEFAULT_PARAMS


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_beeswarm(sv, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_charts_grid(shap_by_level: dict, case_type: str, target: str,
                           path: Path, top_n: int = 12):
    """2x2 grid of mean|SHAP| bar charts, one panel per Model 1-4.

    shap_by_level maps level -> list of (feature, mean_abs_shap) sorted desc.
    Each model has its own feature set, so panels are plotted independently
    rather than as a shared grouped bar chart.
    """
    colors = {"1": "#2980b9", "2": "#27ae60", "3": "#8e44ad", "4": "#e67e22"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for level, ax, label in zip(LEVELS, axes.flat, LEVEL_LABELS):
        feats_vals = shap_by_level[level][:top_n]
        # barh draws bottom-up, so reverse to put the largest bar on top.
        feats = [f for f, _ in feats_vals][::-1]
        vals  = [v for _, v in feats_vals][::-1]

        bars = ax.barh(feats, vals, color=colors[level], alpha=0.85)
        vmax = max(vals) if vals else 1.0
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=8)

        ax.set_title(label.replace("\n", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("mean(|SHAP|)")
        ax.set_xlim(0, vmax * 1.18)
        ax.margins(y=0.01)

    fig.suptitle(f"SHAP feature importance by model | {case_type.upper()} | {target}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(y_true, y_pred, case_type: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=4, color="#2980b9", rasterized=True)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect")
    ax.set_xlabel("Actual event density (ed)")
    ax.set_ylabel("Predicted event density (ed)")
    ax.set_title(f"Actual vs Predicted — Model 4 | {case_type.upper()}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mae_comparison(all_results: dict, path: Path):
    """Grouped bar chart of MAE for Models 1-4 across case types."""
    case_types = list(all_results.keys())
    x          = np.arange(len(LEVELS))
    width      = 0.35
    colors     = ["#2980b9", "#e67e22"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (ct, color) in enumerate(zip(case_types, colors)):
        maes = [all_results[ct]["models"][lv]["mae"] for lv in LEVELS]
        bars = ax.bar(x + i * width, maes, width, label=ct.upper(), color=color, alpha=0.85)
        for bar, v in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("XGBoost: MAE by Model Level and Case Type")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_r2_waterfall(result: dict, path: Path):
    """R² progression across Models 1-4 for one case type."""
    ct     = result["case_type"]
    r2s    = [result["models"][lv]["r2"] for lv in LEVELS]
    labels = LEVEL_LABELS
    colors = ["#2980b9"] + [
        "#27ae60" if r2s[i] >= r2s[i - 1] else "#e74c3c" for i in range(1, len(r2s))
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
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

def run_case_type(df, case_type: str, params: dict, target: str) -> dict:
    print(f"\n── XGBoost | {case_type.upper()} | target={target} ──")
    models_out   = {}
    feature_sets = get_feature_sets(df)
    preds        = {}
    y_test_vals  = None
    shap_by_level = {}

    for level in LEVELS:
        X_train, X_test, y_train, y_test = prepare(df, case_type, level, target=target)

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
        shap_by_level[level] = top15

        FIGURES.mkdir(parents=True, exist_ok=True)
        if level == RICHEST:
            _plot_beeswarm(sv, f"SHAP Beeswarm — Model 4 | {case_type.upper()} | {target}",
                           FIGURES / f"xgb_shap_beeswarm_{target}_{case_type}.png")

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

    _plot_scatter(y_test_vals, preds[RICHEST], case_type,
                  FIGURES / f"xgb_scatter_{target}_{case_type}.png")

    # Combined SHAP importance charts for Models 1-4 in one figure.
    _plot_shap_charts_grid(shap_by_level, case_type, target,
                           FIGURES / f"xgb_shap_charts_{target}_{case_type}.png")

    m = models_out
    result = {
        "case_type": case_type,
        "target":    target,
        "models":    m,
        # MAE reductions isolating each signal (positive = lower error)
        "m2_vs_m1_mae_improvement": round(m["1"]["mae"] - m["2"]["mae"], 6),  # workload alone
        "m3_vs_m1_mae_improvement": round(m["1"]["mae"] - m["3"]["mae"], 6),  # judge ID alone
        "m4_vs_m3_mae_improvement": round(m["3"]["mae"] - m["4"]["mae"], 6),  # workload | judge ID
        "m4_vs_m2_mae_improvement": round(m["2"]["mae"] - m["4"]["mae"], 6),  # judge ID | workload
        "m4_vs_m1_mae_improvement": round(m["1"]["mae"] - m["4"]["mae"], 6),  # both
    }
    print(f"  Δ2-1={result['m2_vs_m1_mae_improvement']:+.4f}  "
          f"Δ3-1={result['m3_vs_m1_mae_improvement']:+.4f}  "
          f"Δ4-3={result['m4_vs_m3_mae_improvement']:+.4f}  "
          f"Δ4-2={result['m4_vs_m2_mae_improvement']:+.4f}  "
          f"Δ4-1={result['m4_vs_m1_mae_improvement']:+.4f}")

    # R² waterfall
    _plot_r2_waterfall(result, FIGURES / f"xgb_r2_waterfall_{target}_{case_type}.png")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr", "both"], default="both")
    args = parser.parse_args()

    params = _load_params(TARGET)
    df     = _as_categorical(load_dataset())
    types  = ["cv", "cr"] if args.case_type == "both" else [args.case_type]

    all_results = {}
    for ct in types:
        all_results[ct] = run_case_type(df, ct, params, TARGET)

    # Cross-case-type MAE comparison chart (only when both are run)
    if len(all_results) > 1:
        FIGURES.mkdir(parents=True, exist_ok=True)
        _plot_mae_comparison(all_results, FIGURES / f"xgb_mae_comparison_{TARGET}.png")
        print(f"\nSaved → {FIGURES / f'xgb_mae_comparison_{TARGET}.png'}")

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"xgb_results_{TARGET}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
