#!/usr/bin/env python3
"""Stream-profile Event Log.csv without loading into memory."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "Event Log.csv"


def profile(path: Path, sample: int | None) -> None:
    activities: Counter[str] = Counter()
    case_types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    ucids: set[str] = set()
    min_d = max_d = None
    n = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("Empty or invalid CSV")
        n_cols = len(reader.fieldnames)
        attr_cols = [c for c in reader.fieldnames if c.startswith("attribute_")]

        for row in reader:
            n += 1
            ucids.add(row["ucid"])
            activities[row["Activity"]] += 1
            case_types[row["case_type"]] += 1
            statuses[row["case_status"]] += 1
            d = row["date_filed"]
            if d:
                min_d = d if min_d is None or d < min_d else min_d
                max_d = d if max_d is None or d > max_d else max_d
            if sample and n >= sample:
                break

    print(f"path: {path}")
    print(f"rows_processed: {n:,}")
    print(f"columns: {n_cols} ({len(attr_cols)} attribute_*)")
    print(f"unique_ucid: {len(ucids):,}")
    if ucids:
        print(f"avg_events_per_case: {n / len(ucids):.1f}")
    print(f"date_range: {min_d} -> {max_d}")
    print("case_type:", dict(case_types))
    print("case_status (top 5):", statuses.most_common(5))
    print("activities (top 12):", activities.most_common(12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--sample", type=int, default=None, help="Max rows (dev quick check)")
    args = p.parse_args()
    if not args.path.is_file():
        raise SystemExit(f"File not found: {args.path}")
    profile(args.path, args.sample)


if __name__ == "__main__":
    main()
