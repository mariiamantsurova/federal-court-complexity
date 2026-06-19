#!/usr/bin/env python3
"""
Compute dynamic judge workload at case open date.

For each case, counts (grouped by District_Judge, on the focal case's filing date):
  - open_cases_at_filing          : cases the same judge had open on the focal
                                    case's filing date (open <= filing < close),
                                    excluding the focal case itself
  - aged_open_cases_at_filing     : subset of the open cases above whose age-so-far
                                    (filing - peer_open) exceeds the 75th percentile
                                    of completed case duration for the *peer's* case
                                    family (cv / cr). Captures how many of a judge's
                                    open cases are relatively "old" for their type.
  - clearance_rate_last_180_days  : cases the same judge closed in the 180 days
                                    before filing divided by cases the judge had
                                    newly assigned in that same window. > 1 means
                                    the docket is shrinking, < 1 means it is growing.
                                    Defined as 0.0 when no new cases were assigned.

The 75th-percentile duration threshold is computed once per case family from
completed case lifetimes (case_close_date - case_open_date) and is static across
filings, so it carries no look-ahead beyond each case's own type.

Requires columns: District_Judge, case_type, case_open_date, case_close_date.

Usage:
  from src.judge_workload import add_judge_workload
  cases = add_judge_workload(cases)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Age percentile that marks an open case as "aged" relative to its case family.
AGED_QUANTILE = 0.75


def add_judge_workload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    open_dates  = pd.to_datetime(df["case_open_date"]).values.astype("datetime64[ns]")
    close_dates = pd.to_datetime(df["case_close_date"]).values.astype("datetime64[ns]")

    # Static 75th-percentile case duration (in days) per case family. Each peer is
    # judged "aged" against the threshold for *its own* case type.
    duration_days = (close_dates - open_dates) / np.timedelta64(1, "D") # type: ignore
    thresh_by_type = (
        pd.Series(duration_days, index=df.index)
        .groupby(df["case_type"], dropna=False)
        .quantile(AGED_QUANTILE)
    )
    peer_thresh = df["case_type"].map(thresh_by_type).to_numpy(dtype=np.float64)

    open_at_filing  = np.zeros(len(df), dtype=np.int32)
    aged_at_filing  = np.zeros(len(df), dtype=np.int32)
    clearance_180   = np.zeros(len(df), dtype=np.float64)

    day = np.timedelta64(1, "D")

    for _, group in df.groupby("District_Judge", dropna=True):
        pos = group.index  # positional integer index into df
        g_open   = open_dates[pos]   # type: ignore
        g_close  = close_dates[pos]  # type: ignore
        g_thresh = peer_thresh[pos]  # type: ignore  per-peer aged threshold (days)

        # broadcast: rows = focal case i, cols = peer case j
        oi = g_open.reshape(-1, 1)   # filing date of focal case
        oj = g_open.reshape(1, -1)   # open date of peer
        cj = g_close.reshape(1, -1)  # close date of peer

        # peer is open at focal filing: opened on/before filing AND not yet closed
        is_open = (oj <= oi) & (cj > oi)

        # open at filing (excluding focal case itself)
        open_at_filing[pos] = (is_open.sum(axis=1) - 1).clip(min=0)

        # aged open: open AND age-so-far exceeds the peer's case-family threshold.
        # The focal case has age 0, so it never counts as aged — no -1 needed.
        peer_age_days = (oi - oj) / day                  # age of peer j at filing i
        is_aged = is_open & (peer_age_days > g_thresh.reshape(1, -1))
        aged_at_filing[pos] = is_aged.sum(axis=1)

        # 180-day window [filing - 180, filing): peers opened (excl. focal) & closed
        opened_180 = (((oi - 180 * day) <= oj) & (oj < oi)).sum(axis=1)
        closed_180 = (((oi - 180 * day) <= cj) & (cj < oi)).sum(axis=1)
        clearance_180[pos] = np.divide(
            closed_180, opened_180,
            out=np.zeros_like(closed_180, dtype=np.float64),
            where=opened_180 > 0,
        )

    df["open_cases_at_filing"]         = open_at_filing
    df["aged_open_cases_at_filing"]    = aged_at_filing
    df["clearance_rate_last_180_days"] = clearance_180
    return df
