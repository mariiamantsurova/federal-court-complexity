"""
Preprocessing for the 4-level model comparison.

Feature sets:
  1 — filing attributes only (no judge info)
  2 — filing attributes + judge workload features (no judge identity)
  3 — filing attributes + District_Judge (judge identity)
  4 — filing attributes + District_Judge + judge workload features

Models 2 and 4 isolate the workload signal with and without judge identity, so
its contribution can be attributed independently of who the judge is.

Usage:
  from src.preprocessing import load_dataset, prepare, get_feature_sets
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge_workload import add_judge_workload

PARQUET_PATH = ROOT / "data" / "by_case.parquet"

# Two count columns are log1p-transformed; clearance is a bounded ratio, left raw.
WORKLOAD_COUNT_COLS = ["open_cases_at_filing", "aged_open_cases_at_filing"]
WORKLOAD_COLS = WORKLOAD_COUNT_COLS + ["clearance_rate_last_180_days"]

# aggregated target, which would be a separate pipeline (see git history).
TARGET = "ed"

_NON_FEATURE = {
    "case_type",        # subsetting only
    "ucid",             # case identifier
    "case_open_date",   # temporal split + workload input
    "case_close_date",  # workload input
    "n_events",         # target building block — leakage
    "n_activity_types", # target building block — leakage
    "year"         # derived from case_open_date
}


def load_dataset(parquet_path=None) -> pd.DataFrame:
    """
    Load by_case.parquet, parse dates, and compute judge workload features on
    the full dataset (District_Judge is the actual judge id; no lookup needed).
    Workload must be computed on ALL cases so peer counts are accurate.
    """
    df = pd.read_parquet(parquet_path or PARQUET_PATH).reset_index()

    df["case_open_date"]  = pd.to_datetime(df["case_open_date"])
    df["case_close_date"] = pd.to_datetime(df["case_close_date"])

    print("Computing judge workload features (all cases)...")
    df = add_judge_workload(df)
    # Counts are right-skewed → log1p. The clearance ratio is already bounded.
    df[WORKLOAD_COUNT_COLS] = np.log1p(df[WORKLOAD_COUNT_COLS])
    print(f"  Done. Workload columns: {WORKLOAD_COLS}")

    return df


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Train on oldest (1-test_ratio) fraction, test on newest test_ratio."""
    sorted_idx = df["case_open_date"].argsort()
    cutoff_pos = int(len(df) * (1 - test_ratio))
    train_idx = df.index[sorted_idx[:cutoff_pos]] # type: ignore
    test_idx  = df.index[sorted_idx[cutoff_pos:]] # type: ignore
    return df.loc[train_idx], df.loc[test_idx]


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return column lists for Models 1-4 (target excluded from X)."""
    all_excluded = (_NON_FEATURE | {TARGET}
                    | set(WORKLOAD_COLS) | {"District_Judge"})
    base_cols = sorted(c for c in df.columns if c not in all_excluded)
    return {
        "1": base_cols,
        "2": base_cols + WORKLOAD_COLS,
        "3": base_cols + ["District_Judge"],
        "4": base_cols + ["District_Judge"] + WORKLOAD_COLS,
    }


def prepare(
    df: pd.DataFrame,
    case_type: str,
    model_level: str,
    target: str = TARGET,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_train, y_test) for a case_type × model level."""
    if target not in df.columns:
        raise KeyError(
            f"target '{target}' not in dataset. "
            "Re-run the notebook's Build & Save cell to regenerate by_case.parquet."
        )

    sub = df[df["case_type"] == case_type].copy()
    train, test = temporal_split(sub, test_ratio)

    cols = get_feature_sets(df)[model_level]
    return train[cols], test[cols], train[target], test[target]
