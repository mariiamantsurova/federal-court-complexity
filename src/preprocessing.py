"""
Preprocessing for the 3-level model comparison.

Feature sets:
  A — filing attributes only (no judge info)
  B — filing attributes + District_Judge_idx (judge identity)
  C — filing attributes + District_Judge_idx + judge workload features

Usage:
  from src.preprocessing import load_dataset, prepare, get_feature_sets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge_workload import add_judge_workload

PARQUET_PATH = ROOT / "data" / "by_case.parquet"
LOOKUP_PATH  = ROOT / "data" / "district_judge_lookup.json"

WORKLOAD_COLS = ["judge_open_at_filing", "judge_opened_30d", "judge_closed_30d"]


# aggregated target, which would be a separate pipeline (see git history).
TARGET = "ed"

_NON_FEATURE = {
    "case_type",        # subsetting only
    "ucid",             # case identifier
    "case_open_date",   # temporal split + workload input
    "case_close_date",  # workload input
    "District_Judge",   # intermediate string; reverse-mapped for workload only
    "n_events",         # target building block — leakage
    "n_activity_types", # target building block — leakage
    "year"         # derived from case_open_date
}


def load_dataset(parquet_path=None, lookup_path=None) -> pd.DataFrame:
    """
    Load by_case.parquet, reverse-map District_Judge_idx → judge string,
    parse dates, and compute judge workload features on the full dataset.
    Workload must be computed on ALL cases so peer counts are accurate.
    """
    df = pd.read_parquet(parquet_path or PARQUET_PATH).reset_index()

    lookup = json.load(open(lookup_path or LOOKUP_PATH))
    df["District_Judge"] = df["District_Judge_idx"].astype(str).map(lookup)

    df["case_open_date"]  = pd.to_datetime(df["case_open_date"])
    df["case_close_date"] = pd.to_datetime(df["case_close_date"])

    print("Computing judge workload features (all cases)...")
    df = add_judge_workload(df)
    df['case_open_date'] = 
    df
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
    """Return column lists for Model A, B, and C (target excluded from X)."""
    all_excluded = (_NON_FEATURE | {TARGET}
                    | set(WORKLOAD_COLS) | {"District_Judge_idx"})
    base_cols = sorted(c for c in df.columns if c not in all_excluded)
    return {
        "A": base_cols,
        "B": base_cols + ["District_Judge_idx"],
        "C": base_cols + ["District_Judge_idx"] + WORKLOAD_COLS,
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
