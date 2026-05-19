#!/usr/bin/env python3
"""
Stream Event Log.csv and build case-level complexity + LOS features.

Output: data/case_features.parquet (one row per ucid)

Usage:
  python src/build_features.py
  python src/build_features.py --sample-rows 200000
  python src/build_features.py --case-type cv --closed-only
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "Event Log.csv"
DEFAULT_OUTPUT = ROOT / "data" / "case_features.parquet"

STATIC_COLS = [
    "case_status",
    "case_type",
    "nature_suit",
    "city",
    "is_mdl",
    "District_Judge",
    "Magistrate_Judge",
    "plaintiffs_count",
    "Defendants_count",
    "plaintiffs_counsels_count",
    "Defendants_counsels_count",
    "plaintiffs_share_pro_se",
    "Defendants_share_pro_se",
    "related_case_count",
]

COMPLEXITY_COLS = [
    "n_events",
    "n_motions",
    "n_orders",
    "n_notices",
    "n_activity_types",
    "activity_entropy",
    "n_attribute_flags",
    "rework_ratio",
    "time_gaps_std",
    "party_load",
    "counsel_load",
    "pro_se_parties",
]


def _parse_date(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _to_float(s: str) -> float:
    try:
        return float(s) if s not in ("", "Missing", None) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _activity_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def _time_gaps_std(dates: list[datetime]) -> float:
    uniq = sorted({d.date() for d in dates})
    if len(uniq) < 2:
        return 0.0
    gaps = [(uniq[i + 1] - uniq[i]).days for i in range(len(uniq) - 1)]
    return float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float(gaps[0])


@dataclass
class CaseAccumulator:
    activities: Counter[str] = field(default_factory=Counter)
    dates: list[datetime] = field(default_factory=list)
    events_chrono: list[tuple[datetime, str]] = field(default_factory=list)
    attr_flags_true: set[str] = field(default_factory=set)
    static: dict[str, str] = field(default_factory=dict)
    n_rows: int = 0

    def add_row(self, row: dict[str, str], attr_cols: list[str]) -> None:
        self.n_rows += 1
        act = (row.get("Activity") or "").strip()
        d = _parse_date(row.get("date_filed", ""))
        if act:
            self.activities[act] += 1
        if d:
            self.dates.append(d)
            if act:
                self.events_chrono.append((d, act))
        for col in attr_cols:
            if row.get(col) in ("True", "true", "1"):
                self.attr_flags_true.add(col)
        if not self.static:
            for col in STATIC_COLS:
                if col in row:
                    self.static[col] = row[col]

    def finalize(self) -> dict:
        n_events = sum(self.activities.values())
        plaintiffs = _to_float(self.static.get("plaintiffs_count", "0"))
        defendants = _to_float(self.static.get("Defendants_count", "0"))
        p_counsel = _to_float(self.static.get("plaintiffs_counsels_count", "0"))
        d_counsel = _to_float(self.static.get("Defendants_counsels_count", "0"))
        p_pro = _to_float(self.static.get("plaintiffs_share_pro_se", "0"))
        d_pro = _to_float(self.static.get("Defendants_share_pro_se", "0"))

        closed = self.static.get("case_status") == "closed"
        duration_days = None
        if self.dates:
            duration_days = (max(self.dates) - min(self.dates)).days

        los_days = duration_days if closed else None
        survival_time_days = duration_days
        event_observed = int(closed)

        chrono = sorted(self.events_chrono, key=lambda x: x[0])
        activity_sequence = " > ".join(a for _, a in chrono[:80])
        if len(chrono) > 80:
            activity_sequence += " > ..."

        rework_extra = sum(c - 1 for c in self.activities.values() if c > 1)
        rework_ratio = rework_extra / n_events if n_events else 0.0

        out = {
            **{k: self.static.get(k) for k in STATIC_COLS},
            "n_events": n_events,
            "n_motions": self.activities.get("motion", 0),
            "n_orders": self.activities.get("order", 0),
            "n_notices": self.activities.get("notice", 0),
            "n_activity_types": len(self.activities),
            "activity_entropy": _activity_entropy(self.activities),
            "n_attribute_flags": len(self.attr_flags_true),
            "rework_ratio": rework_ratio,
            "time_gaps_std": _time_gaps_std(self.dates),
            "party_load": plaintiffs + defendants,
            "counsel_load": p_counsel + d_counsel,
            "pro_se_parties": p_pro + d_pro,
            "los_days": los_days,
            "duration_days": survival_time_days,
            "survival_time_days": survival_time_days,
            "event_observed": event_observed,
            "activity_sequence": activity_sequence,
            "date_open": min(self.dates).strftime("%Y-%m-%d") if self.dates else None,
            "date_last": max(self.dates).strftime("%Y-%m-%d") if self.dates else None,
        }
        return out


def stream_build(
    path: Path,
    *,
    sample_rows: int | None,
    case_type: str | None,
    closed_only: bool,
) -> pd.DataFrame:
    cases: dict[str, CaseAccumulator] = {}
    n_rows = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        attr_cols = [c for c in reader.fieldnames if c.startswith("attribute_") and c != "attribute_duplicates"]

        for row in reader:
            n_rows += 1
            if case_type and row.get("case_type") != case_type:
                continue
            ucid = row["ucid"]
            if ucid not in cases:
                cases[ucid] = CaseAccumulator()
            cases[ucid].add_row(row, attr_cols)

            if sample_rows and n_rows >= sample_rows:
                break

    records = []
    for ucid, acc in cases.items():
        rec = acc.finalize()
        rec["ucid"] = ucid
        if closed_only and rec.get("case_status") != "closed":
            continue
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Winsorize heavy tails (e.g. MDL-style mega-cases) before z-scoring
    for col in COMPLEXITY_COLS:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    for col in COMPLEXITY_COLS:
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std()
            df[f"z_{col}"] = (df[col] - mu) / sigma if sigma and sigma > 0 else 0.0
    z_cols = [f"z_{c}" for c in COMPLEXITY_COLS if f"z_{c}" in df.columns]
    df["complexity_index"] = df[z_cols].mean(axis=1).clip(-3, 3)

    if "District_Judge" in df.columns:
        load = df["District_Judge"].value_counts()
        df["judge_caseload"] = df["District_Judge"].map(load)

    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample-rows", type=int, default=None, help="Stop after N CSV rows (dev)")
    p.add_argument("--case-type", choices=("cv", "cr"), default=None)
    p.add_argument("--closed-only", action="store_true", help="Keep only case_status==closed")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Reading {args.input} ...")
    df = stream_build(
        args.input,
        sample_rows=args.sample_rows,
        case_type=args.case_type,
        closed_only=args.closed_only,
    )
    if df.empty:
        raise SystemExit("No cases produced; check filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    closed = df["los_days"].notna().sum() if "los_days" in df.columns else 0
    print(f"Wrote {len(df):,} cases -> {args.output}")
    print(f"  closed with LOS: {closed:,}")
    print(f"  los_days median: {df['los_days'].median():.0f}")
    print(f"  complexity_index median: {df['complexity_index'].median():.3f}")


if __name__ == "__main__":
    main()
