#!/usr/bin/env python3
"""
Compute dynamic judge workload at case open date.

For each case, counts how many other cases the same judge had open
on the day that case was filed. Uses case_open_date and los_days to
derive each case's close date.

Usage:
  from src.judge_workload import add_judge_workload
  df = add_judge_workload(df)   # adds 'judge_workload_at_open' column
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_judge_workload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'judge_workload_at_open' to df: number of cases the same judge
    had open on the focal case's open date (excluding the case itself).

    Requires columns: District_Judge, case_open_date, los_days.
    Cases with missing los_days are treated as still open.
    """
    df = df.copy()
    open_dates = pd.to_datetime(df["case_open_date"])
    los = df["los_days"].fillna(np.inf)
    close_dates = open_dates + pd.to_timedelta(los.clip(upper=1e6), unit="D")

    workload = np.zeros(len(df), dtype=np.int32)

    for _, group in df.groupby("District_Judge", dropna=True):
        idx = group.index
        g_open = open_dates.loc[idx].values.astype("datetime64[ns]")
        g_close = close_dates.loc[idx].values.astype("datetime64[ns]")

        # Matrix: for each case i (row), count cases j (col) where
        # open[j] <= open[i]  AND  close[j] > open[i]
        open_col = g_open.reshape(-1, 1)
        open_row = g_open.reshape(1, -1)
        close_row = g_close.reshape(1, -1)

        mask = (open_row <= open_col) & (close_row > open_col)
        counts = mask.sum(axis=1) - 1  # subtract self

        for pos, i in enumerate(idx):
            workload[df.index.get_loc(i)] = max(0, int(counts[pos]))

    df["judge_workload_at_open"] = workload
    return df
