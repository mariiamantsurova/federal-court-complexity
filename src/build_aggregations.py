#!/usr/bin/env python3
"""
Build three aggregation datasets from case-level features:

  1. by_case.parquet   — one row per ucid (complexity + LOS)
  2. by_judge.parquet  — one row per District_Judge
  3. by_city.parquet   — one row per city

Requires data/case_features.parquet (run src/build_case_features.py first).

Usage:
  python src/build_aggregations.py
  python src/build_aggregations.py --by-case-type   # also write cv / cr subsets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import COMPLEXITY_CORE, TARGET, VALID_CASE_TYPES, add_derived_columns, filter_by_case_type
DEFAULT_CASE_INPUT = ROOT / "data" / "case_features.parquet"
OUT_DIR = ROOT / "data" / "aggregations"

# Metrics aggregated at judge / city level
PERFORMANCE_COLS = [TARGET, "log_los_days"]
COMPLEXITY_AGG_COLS = COMPLEXITY_CORE + ["complexity_index"]
MIX_COLS = ["case_type"]


def _percentile(s: pd.Series, q: float) -> float:
    return float(s.quantile(q))


def _aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarise closed cases by group_col."""
    closed = df.dropna(subset=[TARGET]).copy()
    closed = closed[closed[TARGET] >= 0]
    closed = add_derived_columns(closed)

    sum_cols = [c for c in closed.columns if c.startswith("sum_attribute_")]
    mean_cols = [c for c in COMPLEXITY_AGG_COLS + sum_cols if c in closed.columns]

    agg_spec: dict[str, tuple[str, str | callable]] = {
        "n_cases": (TARGET, "count"),
        "los_mean": (TARGET, "mean"),
        "los_median": (TARGET, "median"),
        "los_std": (TARGET, "std"),
        "log_los_mean": ("log_los_days", "mean"),
    }
    for col in mean_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")

    grouped = closed.groupby(group_col, dropna=False).agg(**agg_spec).reset_index()
    grouped = grouped.rename(columns={group_col: group_col})

    # Case-type mix (% cv)
    if "case_type" in closed.columns:
        mix = (
            closed.groupby(group_col)["case_type"]
            .apply(lambda s: (s == "cv").mean() * 100)
            .reset_index(name="pct_cv")
        )
        grouped = grouped.merge(mix, on=group_col, how="left")

    # LOS percentiles
    pcts = (
        closed.groupby(group_col)[TARGET]
        .agg(
            los_p25=lambda s: _percentile(s, 0.25),
            los_p75=lambda s: _percentile(s, 0.75),
            los_p90=lambda s: _percentile(s, 0.90),
        )
        .reset_index()
    )
    grouped = grouped.merge(pcts, on=group_col, how="left")

    return grouped.sort_values("n_cases", ascending=False).reset_index(drop=True)


def _write_group(
    cases: pd.DataFrame,
    out_dir: Path,
    *,
    label: str,
) -> None:
    """Write by_case / by_judge / by_city parquet files for one case subset."""
    tag = "" if label == "all" else f"_{label}"
    cases = add_derived_columns(cases)

    by_case_path = out_dir / f"by_case{tag}.parquet"
    cases.to_parquet(by_case_path, index=False)

    by_judge = _aggregate(cases, "District_Judge")
    by_judge_path = out_dir / f"by_judge{tag}.parquet"
    by_judge.to_parquet(by_judge_path, index=False)

    by_city = _aggregate(cases, "city")
    by_city_path = out_dir / f"by_city{tag}.parquet"
    by_city.to_parquet(by_city_path, index=False)

    print(f"[{label}] by_case:  {len(cases):,} rows -> {by_case_path}")
    print(f"[{label}] by_judge: {len(by_judge):,} judges -> {by_judge_path}")
    print(f"[{label}] by_city:  {len(by_city):,} cities -> {by_city_path}")


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
        for ct in VALID_CASE_TYPES:
            subset = filter_by_case_type(cases, ct)
            _write_group(subset, out_dir, label=ct)
            result[f"by_case_{ct}"] = add_derived_columns(subset)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_CASE_INPUT)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--by-case-type",
        action="store_true",
        help="Also write by_case/by_judge/by_city parquet files for cv and cr",
    )
    args = p.parse_args()
    build_all(args.input, args.output_dir, by_case_type=args.by_case_type)


if __name__ == "__main__":
    main()
