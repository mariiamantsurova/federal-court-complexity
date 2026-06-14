#!/usr/bin/env python3
"""
Statistical comparison of XGBoost Model A / B / C predictions.

Loads predictions saved by run_xgb.py (docs/xgb_predictions_{ct}.npz)
and runs Wilcoxon signed-rank tests on paired absolute errors.

Wilcoxon is used instead of a t-test because absolute errors are bounded,
right-skewed, and non-normal — a non-parametric paired test is more appropriate.

H0 for each comparison: the two models have equal median absolute error.

Usage:
  .venv/bin/python3 scripts/compare_models.py              # both case types
  .venv/bin/python3 scripts/compare_models.py --case-type cv

Outputs:
  docs/stat_tests.json
  reports/figures/xgb_error_dist_{ct}.png
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
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOCS    = ROOT / "docs"
FIGURES = ROOT / "reports" / "figures"


def _wilcoxon_compare(errors_a: np.ndarray, errors_b: np.ndarray,
                      name_a: str, name_b: str) -> dict:
    diff = errors_a - errors_b   # positive → a is worse than b
    if np.all(diff == 0):
        return {"p_value": 1.0, "significant_at_05": False,
                "significant_at_01": False, "winner": "equal", "mean_mae_delta": 0.0}
    stat, p = wilcoxon(diff, alternative="two-sided")
    winner = name_b if diff.mean() > 0 else name_a
    return {
        "statistic":         round(float(stat), 4), # type: ignore
        "p_value":           round(float(p), 6), # type: ignore
        "significant_at_05": bool(p < 0.05), # type: ignore
        "significant_at_01": bool(p < 0.01), # type: ignore
        "winner":            winner,
        "mean_mae_delta":    round(float(diff.mean()), 6),   # >0 → B has lower error
    }


def _plot_error_dist(abs_errors: dict[str, np.ndarray],
                     stat_tests: dict, case_type: str, path: Path):
    labels = ["A", "B", "C"]
    data   = [abs_errors[lv] for lv in labels]
    colors = ["#2980b9", "#27ae60", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, positions=range(3), showmedians=True)
    for pc, color in zip(parts["bodies"], colors): # type: ignore
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["Model A\n(filing only)", "Model B\n(+ judge ID)", "Model C\n(+ workload)"]
    )
    ax.set_ylabel("Absolute error")
    ax.set_title(f"Error Distribution by Model Level | {case_type.upper()}")

    # Significance brackets between adjacent pairs
    comparisons = [("B_vs_A", 0, 1), ("C_vs_B", 1, 2)]
    y_top = max(np.percentile(d, 95) for d in data) * 1.05
    for key, xi, xj in comparisons:
        p   = stat_tests[key]["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        mid = (xi + xj) / 2
        ax.plot([xi, xi, xj, xj], [y_top, y_top * 1.02, y_top * 1.02, y_top],
                lw=1, color="grey")
        ax.text(mid, y_top * 1.025, f"p={p:.4f} {sig}",
                ha="center", va="bottom", fontsize=8)
        y_top *= 1.12

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def compare_case_type(case_type: str) -> dict:
    npz_path = DOCS / f"xgb_predictions_{case_type}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found — run scripts/run_xgb.py first."
        )

    data   = np.load(npz_path)
    y_test = data["y_test"]
    preds  = {lv: data[f"pred_{lv}"] for lv in ["A", "B", "C"]}

    abs_errors = {lv: np.abs(y_test - preds[lv]) for lv in ["A", "B", "C"]}

    print(f"\n── Statistical tests | {case_type.upper()} "
          f"(n={len(y_test):,} test cases) ──")

    stat_tests = {
        "B_vs_A": _wilcoxon_compare(abs_errors["A"], abs_errors["B"], "A", "B"),
        "C_vs_B": _wilcoxon_compare(abs_errors["B"], abs_errors["C"], "B", "C"),
        "C_vs_A": _wilcoxon_compare(abs_errors["A"], abs_errors["C"], "A", "C"),
    }

    for key, res in stat_tests.items():
        sig = ("***" if res["p_value"] < 0.001 else
               "**"  if res["p_value"] < 0.01  else
               "*"   if res["p_value"] < 0.05  else "ns")
        print(f"  {key}: p={res['p_value']:.6f} {sig:3s}  "
              f"winner={res['winner']}  Δmae={res['mean_mae_delta']:+.6f}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_error_dist(abs_errors, stat_tests, case_type,
                     FIGURES / f"xgb_error_dist_{case_type}.png")

    return {"case_type": case_type, "n_test": int(len(y_test)),
            "stat_tests": stat_tests}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr", "both"], default="both")
    args = parser.parse_args()

    types = ["cv", "cr"] if args.case_type == "both" else [args.case_type]
    all_results = {}
    for ct in types:
        all_results[ct] = compare_case_type(ct)

    out = DOCS / "stat_tests.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
