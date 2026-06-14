#!/usr/bin/env python3
"""
Compute dynamic judge workload at case open date.

For each case, counts:
  - judge_open_at_filing   : cases the same judge had open on the focal case's
                             filing date (open_date <= filing < close_date),
                             excluding the focal case itself
  - judge_opened_30d       : cases opened by the same judge in the 30 days
                             before the focal case's filing date
  - judge_closed_30d       : cases closed by the same judge in the 30 days
                             before the focal case's filing date

Requires columns: District_Judge, case_open_date, case_close_date.

Usage:
  from src.judge_workload import add_judge_workload
  cases = add_judge_workload(cases)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_judge_workload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    open_dates  = pd.to_datetime(df["case_open_date"]).values.astype("datetime64[ns]")
    close_dates = pd.to_datetime(df["case_close_date"]).values.astype("datetime64[ns]")

    open_at_filing = np.zeros(len(df), dtype=np.int32)
    opened_30d     = np.zeros(len(df), dtype=np.int32)
    closed_30d     = np.zeros(len(df), dtype=np.int32)

    day = np.timedelta64(1, "D")

    for _, group in df.groupby("District_Judge", dropna=True):
        pos = group.index  # positional integer index into df
        g_open  = open_dates[pos] # type: ignore
        g_close = close_dates[pos] # type: ignore

        # broadcast: rows = focal case i, cols = peer case j
        oi = g_open.reshape(-1, 1)   # filing date of focal case
        oj = g_open.reshape(1, -1)   # open date of peer
        cj = g_close.reshape(1, -1)  # close date of peer

        # open at filing: peer opened on or before focal filing AND not yet closed
        open_at_filing[pos] = (((oj <= oi) & (cj > oi)).sum(axis=1) - 1).clip(min=0)

        # opened in last 30 days (excluding focal case itself)
        opened_30d[pos] = (((oi - 30 * day) <= oj) & (oj < oi)).sum(axis=1)

        # closed in last 30 days
        closed_30d[pos] = (((oi - 30 * day) <= cj) & (cj < oi)).sum(axis=1)

    df["judge_open_at_filing"] = open_at_filing
    df["judge_opened_30d"]     = opened_30d
    df["judge_closed_30d"]     = closed_30d
    return df
