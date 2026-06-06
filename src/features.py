#!/usr/bin/env python3
"""
Shared feature definitions and derived-column logic used across the pipeline.

Imported by:
  src/build_aggregations.py
  src/preprocessing_trees.py  (COMPLEXITY_CORE already duplicated there for independence)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET = "los_days"

VALID_CASE_TYPES = ["cv", "cr"]

# Core complexity features (raw event-level aggregates from build_case_features.py)
COMPLEXITY_CORE = [
    "n_events",
    "n_activity_types",
    "n_motions",
    "activity_entropy",
]

# ── Derived columns ───────────────────────────────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed columns that build_aggregations writes into by_case.parquet:

      log_los_days    — natural log of LOS (NaN-safe; -inf for los_days=0 → set NaN)
      complexity_index — z-score composite of the four raw complexity metrics

    Safe to call on a DataFrame that already has these columns (no-op for that column).
    Always returns a copy.
    """
    df = df.copy()

    # log LOS
    if "log_los_days" not in df.columns and TARGET in df.columns:
        with np.errstate(divide="ignore"):
            log_vals = np.log(df[TARGET].astype(float))
        log_vals[np.isinf(log_vals)] = np.nan
        df["log_los_days"] = log_vals

    # complexity_index: mean of z-scores of the four raw complexity metrics
    if "complexity_index" not in df.columns:
        available = [c for c in COMPLEXITY_CORE if c in df.columns]
        if available:
            z_scores = pd.DataFrame(index=df.index)
            for col in available:
                mu = df[col].mean()
                sd = df[col].std()
                z_scores[col] = (df[col] - mu) / sd if sd > 0 else 0.0
            df["complexity_index"] = z_scores.mean(axis=1)

    return df
