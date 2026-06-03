#!/usr/bin/env python3
"""
Causal-style regression: complexity → LOS (performance).

Fits nested OLS models on log(LOS) with increasing controls:
  M1 — complexity only (associational)
  M2 — + case_type (pooled only), city, is_mdl
  M3 — + judge fixed effects (within-judge complexity effect)

Usage:
  python scripts/run_causal_regression.py
  python scripts/run_causal_regression.py --case-type cv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    COMPLEXITY_CORE,
    LOG_TARGET,
    TARGET,
    VALID_CASE_TYPES,
    add_derived_columns,
    apply_data_filters,
    tagged_path,
)

DEFAULT_FEATURES = ROOT / "data" / "aggregations" / "by_case.parquet"
FALLBACK_FEATURES = ROOT / "data" / "case_features.parquet"
FIG_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"

MIN_JUDGE_CASES = 30


def _resolve_features_path(features_path: Path, case_type: str | None) -> Path:
    if case_type in VALID_CASE_TYPES:
        tagged = features_path.parent / f"by_case_{case_type}.parquet"
        if tagged.is_file():
            return tagged
    return features_path


def _prepare_regression_frame(
    df: pd.DataFrame,
    *,
    case_type: str | None,
    exclude_mdl: bool = False,
) -> pd.DataFrame:
    work = apply_data_filters(df, case_type=case_type, exclude_mdl=exclude_mdl)
    work = work.dropna(subset=[TARGET, "District_Judge"]).copy()
    work = work[work[TARGET] >= 0]
    work = add_derived_columns(work)

    for col in COMPLEXITY_CORE:
        if col in work.columns:
            mu = work[col].mean()
            sd = work[col].std(ddof=0) or 1.0
            work[f"z_{col}"] = (work[col] - mu) / sd

    work["z_complexity_index"] = work[
        [f"z_{c}" for c in COMPLEXITY_CORE if f"z_{c}" in work.columns]
    ].mean(axis=1)

    judge_counts = work["District_Judge"].value_counts()
    keep_judges = judge_counts[judge_counts >= MIN_JUDGE_CASES].index
    work = work[work["District_Judge"].isin(keep_judges)].copy()
    return work


def _control_terms(case_type: str | None, *, exclude_mdl: bool = False) -> str:
    parts = ["C(city)"]
    if not exclude_mdl:
        parts.append("C(is_mdl)")
    if case_type is None:
        parts.insert(0, "C(case_type)")
    return " + ".join(parts)


def _extract_coefs(result) -> pd.DataFrame:
    rows = []
    for name in result.params.index:
        if name == "Intercept" or not str(name).startswith("z_"):
            continue
        rows.append({
            "term": name,
            "coef": float(result.params[name]),
            "std_err": float(result.bse[name]),
            "p_value": float(result.pvalues[name]),
            "ci_low": float(result.conf_int().loc[name, 0]),
            "ci_high": float(result.conf_int().loc[name, 1]),
        })
    return pd.DataFrame(rows)


def run_causal_regression(
    features_path: Path,
    *,
    case_type: str | None = None,
    exclude_mdl: bool = False,
) -> dict:
    features_path = _resolve_features_path(features_path, case_type)
    if not features_path.is_file():
        features_path = FALLBACK_FEATURES
    if not features_path.is_file():
        raise FileNotFoundError("Run src/build_aggregations.py first.")

    raw = pd.read_parquet(features_path)
    df = _prepare_regression_frame(raw, case_type=case_type, exclude_mdl=exclude_mdl)
    complexity_terms = " + ".join(f"z_{c}" for c in COMPLEXITY_CORE if f"z_{c}" in df.columns)
    controls = _control_terms(case_type, exclude_mdl=exclude_mdl)

    formulas = {
        "M1_complexity_only": f"{LOG_TARGET} ~ {complexity_terms}",
        "M2_with_case_controls": f"{LOG_TARGET} ~ {complexity_terms} + {controls}",
        "M3_with_judge_fe": f"{LOG_TARGET} ~ {complexity_terms} + {controls} + C(District_Judge)",
    }

    label = case_type or "all"
    title_suffix = f" [{label}]" if label != "all" else ""
    if exclude_mdl:
        title_suffix += " (excl. MDL)"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict = {
        "case_type": label,
        "exclude_mdl": exclude_mdl,
        "sample": "no_mdl" if exclude_mdl else "all_cases",
        "n_cases": int(len(df)),
        "models": {},
    }
    coef_frames: list[pd.DataFrame] = []

    for model_name, formula in formulas.items():
        print(f"Fitting {model_name} ({label})...")
        fit = smf.ols(formula, data=df).fit(cov_type="HC1")

        complexity_coefs = fit.params.filter(like="z_")
        all_results["models"][model_name] = {
            "formula": formula,
            "r2": float(fit.rsquared),
            "r2_adj": float(fit.rsquared_adj),
            "n_obs": int(fit.nobs),
            "complexity_effects": {
                k: {
                    "coef": float(fit.params[k]),
                    "p_value": float(fit.pvalues[k]),
                    "interpretation": "1 SD increase → log(LOS) change",
                }
                for k in complexity_coefs.index
            },
        }

        z_coefs = _extract_coefs(fit)
        z_coefs["model"] = model_name
        coef_frames.append(z_coefs)

    coef_all = pd.concat(coef_frames, ignore_index=True)
    coef_path = tagged_path(DOCS_DIR / "03_causal_regression_coefs.csv", case_type, exclude_mdl=exclude_mdl)
    coef_all.to_csv(coef_path, index=False)

    plot_df = coef_all[coef_all["term"].str.startswith("z_")].copy()
    plot_df["term"] = plot_df["term"].str.replace("z_", "", regex=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(
        data=plot_df,
        x="term",
        y="coef",
        hue="model",
        dodge=0.4,
        ax=ax,
        linestyles="",
        errorbar=None,
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Complexity → log(LOS): OLS coefficients{title_suffix}")
    ax.set_ylabel("coefficient (1 SD increase in complexity feature)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig_path = tagged_path(FIG_DIR / "03_causal_complexity_coefs.png", case_type, exclude_mdl=exclude_mdl)
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(8000, len(df)), random_state=42)
    sns.regplot(
        data=sample,
        x="z_complexity_index",
        y=LOG_TARGET,
        scatter_kws={"alpha": 0.15, "s": 8},
        line_kws={"color": "C1"},
        ax=ax,
    )
    ax.set_xlabel("complexity index (z-scored mean of core metrics)")
    ax.set_ylabel("log(LOS days)")
    ax.set_title(f"Complexity vs performance{title_suffix}")
    fig.tight_layout()
    scatter_path = tagged_path(FIG_DIR / "03_complexity_vs_log_los.png", case_type, exclude_mdl=exclude_mdl)
    fig.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    results_path = tagged_path(DOCS_DIR / "03_causal_regression_results.json", case_type, exclude_mdl=exclude_mdl)
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\ncase_type={label} | cases in regression: {len(df):,}")
    for name, info in all_results["models"].items():
        print(f"  {name}: R²={info['r2']:.4f}")

    print(f"\nSaved -> {results_path}")
    return all_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument(
        "--case-type",
        choices=("all", *VALID_CASE_TYPES),
        default="all",
        help="Model subset: all (pooled), cv (civil), or cr (criminal)",
    )
    p.add_argument(
        "--exclude-mdl",
        action="store_true",
        help="Exclude Multi-District Litigation cases (is_mdl == True)",
    )
    args = p.parse_args()

    ct = None if args.case_type == "all" else args.case_type
    run_causal_regression(args.features, case_type=ct, exclude_mdl=args.exclude_mdl)


if __name__ == "__main__":
    main()
