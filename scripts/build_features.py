#!/usr/bin/env python3
"""
Build model-ready Event Log.csv (chunked stream).

- Keeps **closed** cases only (`case_status == "closed"`); drops open-case events
- Converts Magistrate_Judge to boolean (True = present, False = missing)
- Applies log1p to zero-inflated count columns flagged in notebooks/00_eda.ipynb

Usage:
  python scripts/build_features.py
  python scripts/build_features.py --sample-rows 50000
  python scripts/build_features.py --output "data/event_log_model.csv"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "Event Log.csv"
DEFAULT_OUTPUT = ROOT / "Event Log_model.csv"

# Same drops as notebooks/00_eda.ipynb (modeling EDA)
DEFAULT_DROP_COLS = [
    "Defendants_highest_offense_opening",
    "Defendants_highest_offense_terminated",
]

MAGISTRATE_COL = "Magistrate_Judge"
CASE_STATUS_COL = "case_status"
MISSING_TOKENS = {"", "Missing", "missing", "NA", "N/A", "nan", "None"}

# Count columns flagged for log1p by 00_eda.ipynb _suggest_transform (1M-row case sample).
# Shares (*_share_*) are excluded — bounded 0/1, not log-shaped.
LOG1P_COLS = [
    "plaintiffs_count",
    "plaintiffs_counsels_count",
    "Defendants_count",
    "Defendants_counsels_count",
]


def is_missing_series(s: pd.Series) -> pd.Series:
    mask = s.isna()
    as_str = s.astype(str).str.strip()
    return mask | as_str.isin(MISSING_TOKENS)


def magistrate_to_boolean(chunk: pd.DataFrame) -> pd.DataFrame:
    if MAGISTRATE_COL not in chunk.columns:
        raise ValueError(f"Column not in CSV: {MAGISTRATE_COL}")
    chunk = chunk.copy()
    chunk[MAGISTRATE_COL] = ~is_missing_series(chunk[MAGISTRATE_COL])
    return chunk


def filter_closed_cases(chunk: pd.DataFrame) -> pd.DataFrame:
    if CASE_STATUS_COL not in chunk.columns:
        raise ValueError(f"Column not in CSV: {CASE_STATUS_COL}")
    status = chunk[CASE_STATUS_COL].astype(str).str.strip().str.lower()
    return chunk.loc[status == "closed"]


def apply_log1p(chunk: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """log1p on non-negative numeric counts (in-place column values)."""
    chunk = chunk.copy()
    for col in cols:
        if col not in chunk.columns:
            continue
        x = pd.to_numeric(chunk[col], errors="coerce")
        chunk[col] = np.log1p(x.clip(lower=0))
    return chunk


def build_event_log_chunked(
    input_path: Path,
    output_path: Path,
    *,
    drop_cols: list[str],
    chunksize: int,
    sample_rows: int | None,
    magistrate_boolean: bool,
    log1p_cols: list[str],
    closed_only: bool,
) -> tuple[int, int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rows_written = 0
    rows_read = 0
    rows_dropped_open = 0
    t0 = time.perf_counter()

    reader = pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing = [c for c in drop_cols if c not in header]
    if missing:
        raise ValueError(f"Columns not in CSV header: {missing}")
    keep_cols = [c for c in header if c not in drop_cols]
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Dropping {len(drop_cols)} columns -> {len(keep_cols)} columns remain")
    print(f"Drop: {drop_cols}")
    if magistrate_boolean:
        if MAGISTRATE_COL not in keep_cols:
            raise ValueError(f"{MAGISTRATE_COL} was dropped or missing from input")
        print(f"Convert: {MAGISTRATE_COL} -> boolean (True = present, False = missing)")
    if log1p_cols:
        present = [c for c in log1p_cols if c in keep_cols]
        print(f"log1p: {present}")
    if closed_only:
        if CASE_STATUS_COL not in keep_cols:
            raise ValueError(f"{CASE_STATUS_COL} was dropped or missing from input")
        print(f"Filter: keep {CASE_STATUS_COL} == 'closed' only")

    first_chunk = True
    for chunk in reader:
        if sample_rows is not None:
            remaining = sample_rows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)
        if closed_only:
            n_before = len(chunk)
            chunk = filter_closed_cases(chunk)
            rows_dropped_open += n_before - len(chunk)
            if chunk.empty:
                continue
        out_chunk = chunk.drop(columns=drop_cols, axis=1)
        if magistrate_boolean:
            out_chunk = magistrate_to_boolean(out_chunk)
        if log1p_cols:
            out_chunk = apply_log1p(out_chunk, log1p_cols)
        out_chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        rows_written += len(out_chunk)
        first_chunk = False

        elapsed = time.perf_counter() - t0
        print(f"  ... {rows_written:,} rows written ({elapsed:.0f}s)")

        if sample_rows is not None and rows_read >= sample_rows:
            break

    elapsed = time.perf_counter() - t0
    print(f"Done: {rows_written:,} rows written in {elapsed:.1f}s")
    print(f"  rows read: {rows_read:,}")
    if closed_only:
        print(f"  open-case events skipped: {rows_dropped_open:,}")
    return rows_read, rows_written, rows_dropped_open


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--drop",
        nargs="+",
        default=DEFAULT_DROP_COLS,
        help="Column names to remove",
    )
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--sample-rows", type=int, default=None, help="Stop after N rows (dev)")
    p.add_argument(
        "--no-magistrate-boolean",
        action="store_true",
        help="Keep Magistrate_Judge as original string ids",
    )
    p.add_argument(
        "--no-log1p",
        action="store_true",
        help="Skip log1p on count columns (see LOG1P_COLS in script)",
    )
    p.add_argument(
        "--log1p-cols",
        nargs="*",
        default=None,
        help="Override log1p columns (default: LOG1P_COLS from EDA)",
    )
    p.add_argument(
        "--include-open",
        action="store_true",
        help="Keep open cases (default: closed cases only)",
    )
    args = p.parse_args()

    log1p_cols = [] if args.no_log1p else (args.log1p_cols if args.log1p_cols is not None else LOG1P_COLS)

    build_event_log_chunked(
        args.input,
        args.output,
        drop_cols=args.drop,
        chunksize=args.chunksize,
        sample_rows=args.sample_rows,
        magistrate_boolean=not args.no_magistrate_boolean,
        log1p_cols=log1p_cols,
        closed_only=not args.include_open,
    )


if __name__ == "__main__":
    main()
