#!/usr/bin/env python3
"""Add time-varying concurrent caseload per judge to case_features.parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "case_features.parquet"


def concurrent_load_for_judge(group: pd.DataFrame) -> pd.Series:
    """Count other cases open on same judge with overlapping [date_open, date_last]."""
    g = group.dropna(subset=["date_open", "date_last"]).copy()
    if len(g) == 0:
        return pd.Series(dtype=float)

    starts = pd.to_datetime(g["date_open"]).values.astype("datetime64[D]")
    ends = pd.to_datetime(g["date_last"]).values.astype("datetime64[D]")
    n = len(g)

    # overlap[i,j] = starts[i] <= ends[j] and ends[i] >= starts[j]
    overlap = (starts[:, None] <= ends[None, :]) & (ends[:, None] >= starts[None, :])
    np.fill_diagonal(overlap, False)
    concurrent = overlap.sum(axis=1).astype(float)

    # mean concurrent over case span (approximate via midpoint count)
    mids = starts + (ends - starts) // 2
    mid_overlap = (starts[:, None] <= mids[None, :]) & (ends[:, None] >= mids[None, :])
    np.fill_diagonal(mid_overlap, False)

    out = pd.Series(concurrent, index=g.index, name="judge_concurrent_overlap")
    out_max = pd.Series(mid_overlap.sum(axis=1).astype(float), index=g.index, name="judge_concurrent_at_mid")
    return pd.concat([out, out_max], axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = p.parse_args()

    df = pd.read_parquet(args.path)
    if "District_Judge" not in df.columns:
        raise SystemExit("Missing District_Judge")

    print(f"Computing concurrent load for {len(df):,} cases ...")
    parts = []
    for judge, grp in df.groupby("District_Judge", dropna=False):
        if pd.isna(judge) or judge == "":
            continue
        parts.append(concurrent_load_for_judge(grp))

    loads = pd.concat(parts)
    df["judge_concurrent_overlap"] = loads["judge_concurrent_overlap"]
    df["judge_concurrent_at_mid"] = loads["judge_concurrent_at_mid"]

    df.to_parquet(args.path, index=False)
    print(f"Updated {args.path}")
    print(f"  median concurrent overlap: {df['judge_concurrent_overlap'].median():.0f}")
    print(f"  max concurrent overlap: {df['judge_concurrent_overlap'].max():.0f}")


if __name__ == "__main__":
    main()
