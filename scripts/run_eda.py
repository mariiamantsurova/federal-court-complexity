#!/usr/bin/env python3
"""Generate EDA figures and summary from case_features.parquet."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "case_features.parquet"
FIG_DIR = ROOT / "reports" / "figures"
SUMMARY = ROOT / "docs" / "step3_eda_summary.json"

COMPLEXITY = [
    "n_events", "n_activity_types", "n_motions", "activity_entropy",
    "party_load", "rework_ratio", "complexity_index",
]


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["los_days"].notna()].copy()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1 — LOS distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["los_days"], bins=60, ax=ax)
    ax.set_title("LOS (days) — closed civil cases")
    ax.set_xlabel("los_days")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_los_histogram.png", dpi=120)
    plt.close()

    # 2 — complexity vs LOS
    sample = df.sample(min(8000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=sample, x="complexity_index", y="los_days", alpha=0.25, ax=ax)
    ax.set_title("Complexity index vs LOS")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_complexity_vs_los.png", dpi=120)
    plt.close()

    # 3 — correlation heatmap
    cols = [c for c in COMPLEXITY if c in df.columns] + ["los_days"]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Complexity features vs LOS")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=120)
    plt.close()

    # 4 — LOS by complexity quartile
    df["complexity_quartile"] = pd.qcut(df["complexity_index"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="complexity_quartile", y="los_days", ax=ax)
    ax.set_title("LOS by complexity quartile")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_los_by_complexity_quartile.png", dpi=120)
    plt.close()

    corr_los = corr["los_days"].drop("los_days").sort_values(ascending=False)
    summary = {
        "n_cases": int(len(df)),
        "los_median": float(df["los_days"].median()),
        "los_mean": float(df["los_days"].mean()),
        "complexity_index_median": float(df["complexity_index"].median()),
        "top_correlates_with_los": corr_los.head(5).to_dict(),
        "los_by_quartile_median": df.groupby("complexity_quartile", observed=True)["los_days"].median().to_dict(),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
