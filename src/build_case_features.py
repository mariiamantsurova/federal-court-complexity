#!/usr/bin/env python3
"""
Aggregate Event Log_model.csv to one row per ucid with complexity features + LOS.

Usage:
  python src/build_case_features.py
  python src/build_case_features.py --sample-rows 500000
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "Event Log_model.csv"
DEFAULT_OUTPUT = ROOT / "data" / "case_features.parquet"

STATIC_COLS = [
    "case_type",
    "city",
    "is_mdl",
    "District_Judge",
    "Magistrate_Judge",
    "plaintiffs_count",
    "plaintiffs_share_ind",
    "plaintiffs_share_pro_se",
    "plaintiffs_share_pro_hac_vice",
    "plaintiffs_counsels_count",
    "Defendants_count",
    "Defendants_share_ind",
    "Defendants_share_pro_se",
    "Defendants_share_pro_hac_vice",
    "Defendants_counsels_count",
    "Defendants_pending_counts",
    "Defendants_terminated_counts",
    "Other_courts",
    "related_case_count",
    "Party_Amicus",
    "Party_Counter_Claimant",
    "Party_Counter_Defendant",
    "Party_Court_Monitor",
    "Party_Intervenor",
    "Party_Material_Witness",
    "Party_Third_Party_Defendant",
    "Party_Third_Party_Plaintiff",
    "Party_Trustee",
]


def _activity_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


@dataclass
class CaseAcc:
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    n_events: int = 0
    activities: Counter[str] = field(default_factory=Counter)
    n_motions: int = 0
    attr_sums: Counter[str] = field(default_factory=Counter)
    static: dict[str, object] | None = None


def _update_case(acc: CaseAcc, grp: pd.DataFrame, attr_cols: list[str], static_cols: list[str]) -> None:
    acc.n_events += len(grp)

    dates = pd.to_datetime(grp["date_filed"], errors="coerce")
    dmin, dmax = dates.min(), dates.max()
    if pd.notna(dmin):
        acc.min_date = dmin if acc.min_date is None else min(acc.min_date, dmin)
    if pd.notna(dmax):
        acc.max_date = dmax if acc.max_date is None else max(acc.max_date, dmax)

    acc.activities.update(grp["Activity"].dropna().astype(str).value_counts().to_dict())
    acc.n_motions += int((grp["Activity"] == "motion").sum())

    if attr_cols:
        true_counts = grp[attr_cols].sum(numeric_only=False)
        for col, cnt in true_counts.items():
            if cnt:
                acc.attr_sums[col] += int(cnt)

    if acc.static is None and static_cols:
        acc.static = grp.iloc[0][static_cols].to_dict()


def build_case_features(
    input_path: Path,
    output_path: Path,
    *,
    chunksize: int = 500_000,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    attr_cols = [
        c
        for c in header
        if c.startswith("attribute_") and c != "attribute_duplicates"
    ]
    static_cols = [c for c in STATIC_COLS if c in header]
    usecols = ["ucid", "date_filed", "Activity", *attr_cols, *static_cols]

    cases: dict[str, CaseAcc] = defaultdict(CaseAcc)
    rows_read = 0
    t0 = time.perf_counter()

    for chunk in pd.read_csv(
        input_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        if sample_rows is not None:
            remaining = sample_rows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        for ucid, grp in chunk.groupby("ucid", sort=False):
            _update_case(cases[ucid], grp, attr_cols, static_cols)

        rows_read += len(chunk)
        elapsed = time.perf_counter() - t0
        print(f"  ... {rows_read:,} rows, {len(cases):,} cases ({elapsed:.0f}s)")

        if sample_rows is not None and rows_read >= sample_rows:
            break

    records: list[dict[str, object]] = []
    for ucid, acc in cases.items():
        los_days = None
        if acc.min_date is not None and acc.max_date is not None:
            los_days = (acc.max_date - acc.min_date).days

        rec: dict[str, object] = {
            "ucid": ucid,
            "los_days": los_days,
            "case_open_date": acc.min_date.date().isoformat() if acc.min_date is not None else None,
            "n_events": acc.n_events,
            "n_activity_types": len(acc.activities),
            "n_motions": acc.n_motions,
            "activity_entropy": _activity_entropy(acc.activities),
        }
        if acc.static:
            rec.update(acc.static)
        for col, cnt in acc.attr_sums.items():
            rec[f"sum_{col}"] = cnt
        records.append(rec)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"Done: {len(df):,} cases -> {output_path} ({elapsed:.1f}s)")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--sample-rows", type=int, default=None, help="Stop after N event rows (dev)")
    args = p.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    build_case_features(
        args.input,
        args.output,
        chunksize=args.chunksize,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    main()
