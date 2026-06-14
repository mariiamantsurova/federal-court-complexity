#!/usr/bin/env python3
"""
Preprocessing for tree-based models (Random Forest, XGBoost).

Target: complexity_index (see src/features.py)

Features:
  Model A (include_workload=False):
    - Filing attributes: party counts, shares, party types
    - Suit structure: n_unique_suits, suit_entropy, has_<suit>, ...
    - Case flags: is_cv, is_mdl
    - City (one-hot encoded)

  Model B (include_workload=True):
    - Everything in Model A + judge_workload_at_open

Complexity metrics (n_events, n_activity_types, n_motions, activity_entropy,
complexity_index) are NEVER included in X — they are retrospective.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features import FILING_FEATURES_NUMERIC, PARTY_TYPE_FEATURES, TARGET
from src.suit_features import build_suit_features_dataframe


CATEGORICAL_COLS = ["city"]


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FILING_FEATURES_NUMERIC + PARTY_TYPE_FEATURES if c in df.columns]
    X = df[cols].copy()
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=cols, index=X.index)


def _suit_features(df: pd.DataFrame) -> pd.DataFrame:
    if "nature_suits" not in df.columns:
        return pd.DataFrame(index=df.index)
    return build_suit_features_dataframe(df["nature_suits"])


def _case_flags(df: pd.DataFrame) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    if "case_type" in df.columns:
        cols["is_cv"] = (df["case_type"] == "cv").astype(float)
    if "is_mdl" in df.columns:
        cols["is_mdl_flag"] = df["is_mdl"].astype(float)
    return pd.DataFrame(cols, index=df.index) if cols else pd.DataFrame(index=df.index)


def _city_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not available:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[available], columns=available, drop_first=True, dtype=float)


def prepare_for_trees(
    df: pd.DataFrame,
    target: str = TARGET,
    include_workload: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare case-level data for tree models.

    Args:
        df: DataFrame from data/aggregations/by_case.parquet,
            must already have complexity_index column (from add_derived_columns).
        target: target column (default: complexity_index).
        include_workload: if True, adds judge_workload_at_open as a feature (Model B).

    Returns:
        (X, y) ready for sklearn/xgboost.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame. Run add_derived_columns first.")

    y = df[target].copy()
    valid = y.notna()
    df, y = df[valid].copy(), y[valid].copy()

    X_num  = _numeric_features(df)
    X_suit = _suit_features(df)
    X_flag = _case_flags(df)
    X_city = _city_features(df)

    parts = [X_num, X_suit, X_flag, X_city]

    if include_workload and "judge_workload_at_open" in df.columns:
        wl = df[["judge_workload_at_open"]].copy().fillna(0).astype(float)
        wl.index = X_num.index
        parts.append(wl)

    X = pd.concat(parts, axis=1).reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y
