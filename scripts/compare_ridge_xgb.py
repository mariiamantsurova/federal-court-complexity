#!/usr/bin/env python
"""
Compare Ridge vs XGBoost across Models 1-4 (civil cases) for R^2 and MAE.

Poster-themed, transparent-background figures (no axis labels / titles), so they
drop straight onto the presentation slides:
  - reports/figures/ridge_vs_xgb_r2_ed.png   (R^2, higher is better)
  - reports/figures/ridge_vs_xgb_mae_ed.png  (MAE, lower is better)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT     = Path(__file__).resolve().parents[1]
DOCS     = ROOT / "docs"
FIG_DIR  = ROOT / "reports" / "figures"

LEVELS = ["1", "2", "3", "4"]
NAMES  = ["Case basics", "+ workload", "+ the judge", "+ both"]

# ── Presentation theme (matches scripts/poster_visuals.py) ────────────────────
FONT      = "Avenir Next"
INK       = "#1d2433"
ink_muted = "#7a6f5b"
RIDGE     = "#4c60a7"  # slate
XGB       = "#5C8467"   # olive
SPINE     = "#cbbfa6"

sns.set_theme(style="white")
plt.rcParams.update({
    "font.family": FONT, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": SPINE,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def _load():
    ridge = json.loads((DOCS / "ridge_results_ed.json").read_text())
    xgb   = json.loads((DOCS / "xgb_results_ed.json").read_text())
    return ridge, xgb


def _make_figure(ridge_vals, xgb_vals, fmt, out_name,
                 ridge_color=RIDGE, xgb_color=XGB, legend_loc="lower right"):
    legend_on_top = legend_loc == "top"
    fig = plt.figure(figsize=(11, 6.4))
    ax  = fig.add_axes([0.04, 0.10, 0.92, 0.78 if legend_on_top else 0.84]) # type: ignore
    ax.patch.set_alpha(0)

    x     = np.arange(len(LEVELS))
    width = 0.40
    vmax  = max(ridge_vals + xgb_vals)

    for i, (vals, color, lab) in enumerate(
            [(ridge_vals, ridge_color, "Ridge"), (xgb_vals, xgb_color, "XGBoost")]):
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=lab,
                      zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + vmax * 0.012, fmt.format(v),
                    ha="center", fontsize=12, weight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, fontsize=13)
    ax.set_yticks([])
    upper = legend_loc in ("upper right", "upper left")
    ax.set_ylim(0, vmax * (1.24 if upper else 1.15))
    if legend_on_top:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  frameon=False, fontsize=12.5)
    elif upper:
        ax.legend(loc="upper " + legend_loc.split()[1],
                  bbox_to_anchor=(1.0 if "right" in legend_loc else 0.0, 1.08),
                  frameon=False, fontsize=12.5)
    else:
        ax.legend(loc=legend_loc, frameon=False, fontsize=12.5)

    sns.despine(ax=ax, left=True)
    ax.spines["bottom"].set_color(SPINE)

    fig.savefig(FIG_DIR / out_name, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {FIG_DIR / out_name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ridge, xgb = _load()

    # cv keeps its original filenames; cr adds a _cr suffix.
    for ct, suffix in [("cv", ""), ("cr", "_cr")]:
        if ct not in ridge or ct not in xgb:
            print(f"skipping {ct}: missing from ridge or xgb results")
            continue
        r2_ridge  = [ridge[ct]["models"][lv]["r2"]  for lv in LEVELS]
        r2_xgb    = [xgb[ct]["models"][lv]["r2"]    for lv in LEVELS]
        mae_ridge = [ridge[ct]["models"][lv]["mae"] for lv in LEVELS]
        mae_xgb   = [xgb[ct]["models"][lv]["mae"]   for lv in LEVELS]

        _make_figure(r2_ridge, r2_xgb, "{:.3f}",
                     out_name=f"ridge_vs_xgb_r2{suffix}_ed.png",
                     ridge_color="#4c60a7", xgb_color="#5C8467",
                     legend_loc="upper right")
        _make_figure(mae_ridge, mae_xgb, "{:.3f}",
                     out_name=f"ridge_vs_xgb_mae{suffix}_ed.png",
                     legend_loc="upper right")


if __name__ == "__main__":
    main()
