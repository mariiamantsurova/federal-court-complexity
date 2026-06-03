#!/usr/bin/env python3
"""
Step 2–4 — Survival analysis: Kaplan–Meier curves + Cox proportional hazards.

  duration_days  — follow-up time (days)
  event_observed — 1 = case closed, 0 = censored (still open)

Usage:
  python scripts/run_survival_analysis.py
  python scripts/run_survival_analysis.py --case-type cv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import COMPLEXITY_CORE, VALID_CASE_TYPES, apply_data_filters  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "survival_cases.parquet"
FIG_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"
TABLES_DIR = DOCS_DIR / "tables"

DURATION_COL = "duration_days"
EVENT_COL = "event_observed"


def _prepare_cox_frame(df: pd.DataFrame, *, case_type: str | None) -> pd.DataFrame:
    work = apply_data_filters(df, case_type=case_type, exclude_mdl=False)
    work = work.dropna(subset=[DURATION_COL, EVENT_COL]).copy()
    work[DURATION_COL] = work[DURATION_COL].clip(lower=0)
    work = work[work[DURATION_COL] >= 0]
    # lifelines requires strictly positive durations
    work[DURATION_COL] = work[DURATION_COL].replace(0, 0.5)

    for col in COMPLEXITY_CORE:
        if col in work.columns:
            mu = work[col].mean()
            sd = work[col].std(ddof=0) or 1.0
            work[f"z_{col}"] = (work[col] - mu) / sd

    return work


def _km_plot(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    out_path: Path,
    max_days: int = 2000,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    kmf = KaplanMeierFitter()
    for name, grp in df.groupby(group_col, sort=False, observed=True):
        if len(grp) < 20:
            continue
        kmf.fit(
            grp[DURATION_COL],
            event_observed=grp[EVENT_COL],
            label=f"{name} (n={len(grp):,})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlim(0, max_days)
    ax.set_xlabel("Days since first event")
    ax.set_ylabel("Survival probability (case still open)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _non_constant_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    keep: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.nunique() > 1 and (s.std(ddof=0) or 0) > 0:
            keep.append(col)
    return keep


def _run_cox(df: pd.DataFrame, formula_cols: list[str], *, label: str) -> dict | None:
    formula_cols = _non_constant_cols(df, formula_cols)
    cols = [DURATION_COL, EVENT_COL, *formula_cols]
    cox_df = df[cols].dropna().copy()
    if len(cox_df) < 100 or not formula_cols:
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    except Exception as exc:
        print(f"  {label}: skipped ({exc.__class__.__name__})")
        return None
    summary = cph.summary[["coef", "exp(coef)", "p"]].reset_index()
    summary = summary.rename(columns={"covariate": "term", "index": "term"})
    return {
        "label": label,
        "n_cases": int(len(cox_df)),
        "n_events": int(cox_df[EVENT_COL].sum()),
        "concordance": float(cph.concordance_index_),
        "coefficients": summary.to_dict(orient="records"),
    }


def run_survival_analysis(
    input_path: Path,
    *,
    case_type: str | None = None,
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"{input_path} not found. Run: python src/build_survival_dataset.py"
        )

    raw = pd.read_parquet(input_path)
    df = _prepare_cox_frame(raw, case_type=case_type)
    label = case_type or "all"
    suffix = f"_{label}" if label != "all" else ""

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 2 summary ---
    n = len(df)
    n_events = int(df[EVENT_COL].sum())
    summary = {
        "case_type": label,
        "n_cases": n,
        "n_closed": n_events,
        "n_censored": n - n_events,
        "pct_censored": round(100 * (n - n_events) / n, 2),
        "median_duration_days": float(df[DURATION_COL].median()),
    }
    print(f"\n=== Survival sample [{label}] ===")
    print(f"  cases: {n:,} | closed: {n_events:,} | censored: {n - n_events:,} ({summary['pct_censored']}%)")

    # --- Step 3: Kaplan–Meier ---
    if "case_type" in df.columns and case_type is None:
        _km_plot(
            df,
            group_col="case_type",
            title="Kaplan–Meier survival by case type (cv vs cr)",
            out_path=FIG_DIR / "04_km_by_case_type.png",
        )
        cv = df[df["case_type"] == "cv"]
        cr = df[df["case_type"] == "cr"]
        if len(cv) > 50 and len(cr) > 50:
            lr = logrank_test(
                cv[DURATION_COL], cr[DURATION_COL],
                event_observed_A=cv[EVENT_COL],
                event_observed_B=cr[EVENT_COL],
            )
            summary["logrank_cv_vs_cr_p"] = float(lr.p_value)
            print(f"  log-rank test (cv vs cr): p={lr.p_value:.2e}")

    # complexity quartile KM (pooled or within case_type)
    z_cols = [f"z_{c}" for c in COMPLEXITY_CORE if f"z_{c}" in df.columns]
    if z_cols:
        df = df.copy()
        df["complexity_quartile"] = pd.qcut(
            df[z_cols].mean(axis=1),
            q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )
        _km_plot(
            df,
            group_col="complexity_quartile",
            title=f"Kaplan–Meier by complexity quartile [{label}]",
            out_path=FIG_DIR / f"04_km_by_complexity{suffix}.png",
        )

    # --- Step 4: Cox models ---
    complexity_cols = [f"z_{c}" for c in COMPLEXITY_CORE if f"z_{c}" in df.columns]
    cox_results: list[dict] = []

    m1 = _run_cox(df, complexity_cols, label="Cox_M1_complexity")
    if m1:
        cox_results.append(m1)
        print(f"\n  Cox M1 concordance: {m1['concordance']:.3f}")

    m2_cols = list(complexity_cols)
    if case_type is None and "case_type" in df.columns:
        df = pd.get_dummies(df, columns=["case_type"], drop_first=True, dtype=float)
        m2_cols += [c for c in df.columns if c.startswith("case_type_")]
    if "is_mdl" in df.columns:
        df["is_mdl"] = df["is_mdl"].astype(float)
        m2_cols.append("is_mdl")

    m2 = _run_cox(df, m2_cols, label="Cox_M2_with_controls")
    if m2:
        cox_results.append(m2)
        print(f"  Cox M2 concordance: {m2['concordance']:.3f}")

    # Save outputs
    results = {"summary": summary, "cox_models": cox_results}
    out_json = DOCS_DIR / f"04_survival_results{suffix}.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    if cox_results:
        rows = []
        for model in cox_results:
            for coef in model["coefficients"]:
                rows.append({
                    "case_type": label,
                    "model": model["label"],
                    "term": coef["term"],
                    "coef": round(float(coef["coef"]), 4),
                    "hazard_ratio": round(float(coef["exp(coef)"]), 4),
                    "p_value": float(coef["p"]),
                    "concordance": model["concordance"],
                })
        cox_path = TABLES_DIR / f"T6_survival_cox{suffix}.csv"
        pd.DataFrame(rows).to_csv(cox_path, index=False)
        print(f"\nSaved -> {out_json}, {cox_path}")
    else:
        print(f"\nSaved -> {out_json}")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--case-type",
        choices=("all", *VALID_CASE_TYPES),
        default="all",
    )
    args = p.parse_args()
    ct = None if args.case_type == "all" else args.case_type
    run_survival_analysis(args.input, case_type=ct)


if __name__ == "__main__":
    main()
