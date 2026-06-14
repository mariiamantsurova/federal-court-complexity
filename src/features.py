#!/usr/bin/env python3
"""
Shared feature definitions used across the pipeline.

Research question: does dynamic judge workload at case opening predict
procedural complexity, beyond basic case filing attributes?

  Model A — filing features only (FILING_FEATURES_NUMERIC + PARTY_TYPE_FEATURES + suit + flags)
  Model B — same + judge_workload_at_open

Target: complexity_index (z-score composite of the four complexity metrics).
The complexity metrics themselves are NEVER input features — they are retrospective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "complexity_index"

VALID_CASE_TYPES = ["cv", "cr"]

# Retrospective complexity metrics — these become the TARGET, never features in X.
COMPLEXITY_METRICS = [
    "n_events",
    "n_activity_types",
    "n_motions",
    "activity_entropy",
    "complexity_index",
]

# Features available at case filing (known before any docket activity).
FILING_FEATURES_NUMERIC = [
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
    "Magistrate_Judge",
]

PARTY_TYPE_FEATURES = [
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


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add complexity_index (z-score composite) and log_los_days.
    Safe to call if columns already exist. Always returns a copy.
    """
    df = df.copy()

    if "log_los_days" not in df.columns and "los_days" in df.columns:
        with np.errstate(divide="ignore"):
            log_vals = np.log(df["los_days"].astype(float))
        log_vals[np.isinf(log_vals)] = np.nan
        df["log_los_days"] = log_vals

    if "complexity_index" not in df.columns:
        available = [c for c in ["n_events", "n_activity_types", "n_motions", "activity_entropy"]
                     if c in df.columns]
        if available:
            z = pd.DataFrame(index=df.index)
            for col in available:
                mu, sd = df[col].mean(), df[col].std()
                z[col] = (df[col] - mu) / sd if sd > 0 else 0.0
            df["complexity_index"] = z.mean(axis=1)

    return df
