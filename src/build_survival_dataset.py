#!/usr/bin/env python3
"""
Step 1 — Build survival dataset from raw Event Log.csv (open + closed cases).

Per case (ucid):
  duration_days  — first → last date_filed (follow-up time)
  event_observed — 1 if case_status == closed, 0 if still open (censored)
  complexity features — same core metrics as case-level pipeline

Usage:
  python src/build_survival_dataset.py
  python src/build_survival_dataset.py --sample-rows 500000
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "Event Log.csv"
DEFAULT_OUTPUT = ROOT / "data" / "survival_cases.parquet"

STATIC_COLS = [
    "case_status",
    "case_type",
    "city",
    "is_mdl",
    "District_Judge",
    "Magistrate_Judge",
    "plaintiffs_count",
    "Defendants_count",
    "related_case_count",
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


def _is_closed(status: object) -> bool:
    return str(status or "").strip().lower() == "closed"


@dataclass
class CaseAcc:
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    n_events: int = 0
    activities: Counter[str] = field(default_factory=Counter)
    n_motions: int = 0
    static: dict[str, object] | None = None


def _update_case(acc: CaseAcc, grp: pd.DataFrame, static_cols: list[str]) -> None:
    acc.n_events += len(grp)
    dates = pd.to_datetime(grp["date_filed"], errors="coerce")
    dmin, dmax = dates.min(), dates.max()
    if pd.notna(dmin):
        acc.min_date = dmin if acc.min_date is None else min(acc.min_date, dmin)
    if pd.notna(dmax):
        acc.max_date = dmax if acc.max_date is None else max(acc.max_date, dmax)
    acc.activities.update(grp["Activity"].dropna().astype(str).value_counts().to_dict())
    acc.n_motions += int((grp["Activity"] == "motion").sum())
    if acc.static is None and static_cols:
        acc.static = grp.iloc[0][static_cols].to_dict()


def build_survival_dataset(
    input_path: Path,
    output_path: Path,
    *,
    chunksize: int = 500_000,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    static_cols = [c for c in STATIC_COLS if c in header]
    usecols = ["ucid", "date_filed", "Activity", *static_cols]

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
            _update_case(cases[ucid], grp, static_cols)

        rows_read += len(chunk)
        print(f"  ... {rows_read:,} rows, {len(cases):,} cases ({time.perf_counter() - t0:.0f}s)")
        if sample_rows is not None and rows_read >= sample_rows:
            break

    records: list[dict[str, object]] = []
    for ucid, acc in cases.items():
        duration = None
        if acc.min_date is not None and acc.max_date is not None:
            duration = max((acc.max_date - acc.min_date).days, 0)

        status = (acc.static or {}).get("case_status")
        rec: dict[str, object] = {
            "ucid": ucid,
            "duration_days": duration,
            "event_observed": int(_is_closed(status)),
            "case_open_date": acc.min_date.date().isoformat() if acc.min_date is not None else None,
            "n_events": acc.n_events,
            "n_activity_types": len(acc.activities),
            "n_motions": acc.n_motions,
            "activity_entropy": _activity_entropy(acc.activities),
        }
        if acc.static:
            rec.update(acc.static)
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.dropna(subset=["duration_days"])
    df = df[df["duration_days"] >= 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n = len(df)
    n_closed = int(df["event_observed"].sum())
    print(f"\nDone: {n:,} cases -> {output_path} ({time.perf_counter() - t0:.1f}s)")
    print(f"  closed (event=1): {n_closed:,} ({100 * n_closed / n:.1f}%)")
    print(f"  open/censored:    {n - n_closed:,} ({100 * (n - n_closed) / n:.1f}%)")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--sample-rows", type=int, default=None)
    args = p.parse_args()
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    build_survival_dataset(
        args.input,
        args.output,
        chunksize=args.chunksize,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    main()
