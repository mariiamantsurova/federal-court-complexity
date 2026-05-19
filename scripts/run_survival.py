#!/usr/bin/env python3
"""Kaplan-Meier and Cox PH survival analysis for case duration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "case_features.parquet"
OUT = ROOT / "docs" / "step7_survival.json"
FIG = ROOT / "reports" / "figures"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["case_type"] == "cv"].copy()
    df = df[df["survival_time_days"].notna() & (df["survival_time_days"] >= 0)].copy()
    df["T"] = df["survival_time_days"].astype(float)
    df["E"] = df["event_observed"].astype(int)
    top_suit = df["nature_suit"].value_counts().head(10).index
    df["nature_suit_top"] = df["nature_suit"].where(df["nature_suit"].isin(top_suit), "Other")
    df["complexity_quartile"] = pd.qcut(
        df["complexity_index"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    return df


def run_km(df: pd.DataFrame) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df["complexity_quartile"] == q]
        if len(sub) < 30:
            continue
        kmf.fit(sub["T"], sub["E"], label=str(q))
        kmf.plot_survival_function(ax=ax)
        groups[q] = sub

    ax.set_title("Kaplan-Meier survival by complexity quartile (cv cases)")
    ax.set_xlabel("Days since first event")
    ax.set_ylabel("Survival (not yet closed)")
    fig.tight_layout()
    fig.savefig(FIG / "09_km_by_complexity.png", dpi=120)
    plt.close()

    lr = multivariate_logrank_test(
        df["T"], df["complexity_quartile"].astype(str), df["E"]
    )
    medians = {}
    for q, sub in groups.items():
        km = KaplanMeierFitter()
        km.fit(sub["T"], sub["E"])
        medians[str(q)] = float(km.median_survival_time_ or 0)

    return {
        "logrank_p": float(lr.p_value),
        "median_survival_days_by_quartile": medians,
        "n_cases": int(len(df)),
        "n_events_closed": int(df["E"].sum()),
        "n_censored_open": int((1 - df["E"]).sum()),
    }


def run_cox(df: pd.DataFrame) -> dict:
    use = df[
        [
            "T",
            "E",
            "complexity_index",
            "judge_concurrent_overlap",
            "judge_caseload",
            "party_load",
        ]
    ].dropna()
    use = use[use["T"] > 0]

    cph = CoxPHFitter()
    cph.fit(use, duration_col="T", event_col="E")
    cph.summary.to_csv(ROOT / "docs" / "step7_cox_summary.csv")

    # partial effects plot data in json
    hr = cph.summary[["exp(coef)", "p"]].to_dict()
    return {
        "n_cases": int(len(use)),
        "concordance_index": float(cph.concordance_index_),
        "hazard_ratios": {
            k: {"HR": float(hr["exp(coef)"][k]), "p": float(hr["p"][k])}
            for k in hr["exp(coef)"]
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=PARQUET)
    args = p.parse_args()

    df = pd.read_parquet(args.path)
    if "judge_concurrent_overlap" not in df.columns:
        raise SystemExit("Missing judge_concurrent_overlap; run add_judge_workload.py")

    df = prep(df)
    km = run_km(df)
    cox = run_cox(df)

    out = {"kaplan_meier": km, "cox_ph": cox}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
