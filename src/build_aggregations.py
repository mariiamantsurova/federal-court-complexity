#!/usr/bin/env python3
"""
Build aggregation datasets from case-level features:

  1. by_case.parquet   — one row per case (ucid) with complexity features and LOS
  2. by_judge.parquet  — one row per District Judge summarizing assigned cases

Requires data/case_features.parquet
(run src/build_case_features.py first).

Usage:
  python src/build_aggregations.py
  python src/build_aggregations.py --by-case-type
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (
    COMPLEXITY_CORE,
    TARGET,
    VALID_CASE_TYPES,
    add_derived_columns,
    filter_by_case_type,
)

DEFAULT_CASE_INPUT = ROOT / "data" / "case_features.parquet"
OUT_DIR = ROOT / "data" / "aggregations"

# Metrics averaged at judge level
COMPLEXITY_AGG_COLS = COMPLEXITY_CORE + ["complexity_index"]


def _percentile(q: float) -> Callable[[pd.Series], float]:
    def percentile_func(s: pd.Series) -> float:
        return float(s.quantile(q))

    percentile_func.__name__ = f"p{int(q * 100)}"
    return percentile_func


def _aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate case-level records by a grouping variable
    and compute workload, complexity, and LOS summary statistics.
    """
    closed = df.dropna(subset=[TARGET]).copy()
    closed = closed[closed[TARGET] >= 0]
    closed = add_derived_columns(closed)

    sum_cols = [c for c in closed.columns if c.startswith("sum_attribute_")]
    mean_cols = [c for c in COMPLEXITY_AGG_COLS + sum_cols if c in closed.columns]

    agg_spec: dict[str, tuple[str, str | Callable[[pd.Series], float]]] = {
        "n_cases": (TARGET, "count"),
        "los_mean": (TARGET, "mean"),
        "los_median": (TARGET, "median"),
        "los_std": (TARGET, "std"),
        "los_min": (TARGET, "min"),
        "los_max": (TARGET, "max"),
        "los_p25": (TARGET, _percentile(0.25)),
        "los_p75": (TARGET, _percentile(0.75)),
        "los_p90": (TARGET, _percentile(0.90)),
        "log_los_mean": ("log_los_days", "mean"),
    }

    for col in mean_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")

    if "complexity_index" in closed.columns:
        agg_spec["complexity_index_median"] = ("complexity_index", "median")

    grouped = closed.groupby(group_col, dropna=False).agg(**agg_spec).reset_index()

    if "case_type" in closed.columns:
        mix = (
            closed.groupby(group_col, dropna=False)["case_type"]
            .apply(lambda s: (s == "cv").mean() * 100)
            .reset_index(name="pct_cv")
        )
        grouped = grouped.merge(mix, on=group_col, how="left")

    return grouped.sort_values("n_cases", ascending=False).reset_index(drop=True)


def _write_group(
    cases: pd.DataFrame,
    out_dir: Path,
    *,
    label: str,
) -> None:
    """
    Write by_case and by_judge parquet files for one case subset.
    """
    tag = "" if label == "all" else f"_{label}"
    cases = add_derived_columns(cases)

    by_case_path = out_dir / f"by_case{tag}.parquet"
    cases.to_parquet(by_case_path, index=False)

    by_judge = _aggregate(cases, "District_Judge")
    by_judge_path = out_dir / f"by_judge{tag}.parquet"
    by_judge.to_parquet(by_judge_path, index=False)

    print(f"[{label}] by_case:  {len(cases):,} rows -> {by_case_path}")
    print(f"[{label}] by_judge: {len(by_judge):,} judges -> {by_judge_path}")


def build_all(
    case_input: Path,
    out_dir: Path,
    *,
    by_case_type: bool = False,
) -> dict[str, pd.DataFrame]:
    if not case_input.is_file():
        raise FileNotFoundError(
            f"{case_input} not found. Run: python src/build_case_features.py"
        )

    cases = pd.read_parquet(case_input)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}

    _write_group(cases, out_dir, label="all")
    result["by_case"] = add_derived_columns(cases)

    if by_case_type:
        for case_type in VALID_CASE_TYPES:
            subset = filter_by_case_type(cases, case_type)
            _write_group(subset, out_dir, label=case_type)
            result[f"by_case_{case_type}"] = add_derived_columns(subset)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CASE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--by-case-type",
        action="store_true",
        help="Also write by_case and by_judge parquet files for cv and cr subsets",
    )

    args = parser.parse_args()

    build_all(
        args.input,
        args.output_dir,
        by_case_type=args.by_case_type,
    )


if __name__ == "__main__":
    main()