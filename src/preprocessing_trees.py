#!/usr/bin/env python3
"""
Preprocessing for tree-based models (Random Forest, XGBoost).

Features:
  - Core complexity metrics: n_events, n_activity_types, n_motions, activity_entropy
  - Case attributes: plaintiffs_count, Defendants_count, party counts, etc.
  - Suit structure: n_unique_suits, suit_entropy, has_contract, has_ip, ...
  - City (one-hot encoded)

Excluded:
  - case_type (target of case_type split models)
  - District_Judge (high cardinality, save for judge-level modeling)
  - nature_suit (raw categorical, collinear with case_type)
    BUT INCLUDE suit structure features (derived from nature_suits array)

Usage:
  X, y = prepare_for_trees(df)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.suit_features import build_suit_features_dataframe, get_suit_feature_names


# Core complexity features (always include)
COMPLEXITY_CORE = [
    "n_events",
    "n_activity_types",
    "n_motions",
    "activity_entropy",
    "complexity_index",
]

# Case/party attributes (numeric)
CASE_ATTRIBUTES = [
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
    "Magistrate_Judge",  # Will be treated as numeric (boolean)
]

# Party type counts (all numeric)
PARTY_TYPES = [
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

# Categorical (only city, one-hot encoded)
CATEGORICAL_COLS = ["city"]

# Excluded features
EXCLUDED_COLS = [
    "ucid",
    "case_open_date",
    "case_type",     # Handled explicitly via _prepare_case_flags (binary is_cv feature)
    "District_Judge",  # Handled explicitly via _prepare_judge_feature (target encoding)
    "is_mdl",        # Handled explicitly via _prepare_case_flags
    "los_days",      # Target variable
    "log_los_days",  # Target variable
    "nature_suit",   # Raw categorical, collinear with case_type
    "nature_suits",  # Raw array (we extract features from it)
]

# Sum attribute columns (event type aggregations) - optional, sparse
# Include some key ones, exclude very sparse ones
KEY_ATTRIBUTES = [
    "sum_attribute_scheduling",
    "sum_attribute_hearing_conf",
    "sum_attribute_dismissal_other",
    "sum_attribute_dispositive",
    "sum_attribute_opening",
]


def _get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns to include in model."""
    candidates = (
        COMPLEXITY_CORE
        + CASE_ATTRIBUTES
        + PARTY_TYPES
        + KEY_ATTRIBUTES
    )
    # Only return columns that exist in dataframe
    return [col for col in candidates if col in df.columns]


def _prepare_numeric_features(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Prepare numeric features: select, impute, no scaling (trees don't need it).

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names to use

    Returns:
        DataFrame with selected numeric features, imputed
    """
    X_numeric = df[numeric_cols].copy()

    # Impute missing values (median for numeric)
    imputer = SimpleImputer(strategy="median")
    X_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(X_numeric),
        columns=numeric_cols,
        index=X_numeric.index
    )

    return X_numeric_imputed


def _prepare_judge_feature(
    df: pd.DataFrame,
    judge_target_map: dict | None = None,
) -> pd.DataFrame:
    """
    Encode District_Judge.

    If judge_target_map is provided (built from training-set median log_los_days),
    uses target encoding — a single continuous feature directly correlated with LOS.
    Falls back to label encoding (arbitrary integers) when no map is given.
    """
    if "District_Judge" not in df.columns:
        return pd.DataFrame(index=df.index)
    if judge_target_map is not None:
        fallback = float(np.median(list(judge_target_map.values())))
        encoded = df["District_Judge"].map(judge_target_map).fillna(fallback)
        return pd.DataFrame({"judge_target_encoded": encoded.values.astype(float)}, index=df.index)
    codes, _ = pd.factorize(df["District_Judge"].fillna("<UNK>"))
    return pd.DataFrame({"judge_encoded": codes.astype(float)}, index=df.index)


def _prepare_case_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary features for case_type (is_cv: cv=1, cr=0) and is_mdl."""
    cols: dict[str, pd.Series] = {}
    if "case_type" in df.columns:
        cols["is_cv"] = (df["case_type"] == "cv").astype(float)
    if "is_mdl" in df.columns:
        cols["is_mdl_flag"] = df["is_mdl"].astype(float)
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(cols, index=df.index)


def _prepare_suit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract suit structure features from nature_suits array.

    Args:
        df: Input DataFrame (must have 'nature_suits' column)

    Returns:
        DataFrame with suit feature columns
    """
    if "nature_suits" not in df.columns:
        print("WARNING: 'nature_suits' column not found, skipping suit features")
        return pd.DataFrame(index=df.index)

    suit_features_df = build_suit_features_dataframe(df["nature_suits"])
    return suit_features_df


def _prepare_categorical_features(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """
    Prepare categorical features: one-hot encode, no scaling.

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names

    Returns:
        One-hot encoded DataFrame
    """
    if not categorical_cols:
        return pd.DataFrame(index=df.index)

    # Select only columns that exist
    available_cols = [col for col in categorical_cols if col in df.columns]

    if not available_cols:
        return pd.DataFrame(index=df.index)

    X_cat = df[available_cols].copy()

    # One-hot encode (drop first to avoid multicollinearity)
    X_cat_encoded = pd.get_dummies(
        X_cat,
        columns=available_cols,
        drop_first=True,
        dtype=float
    )

    return X_cat_encoded


def prepare_for_trees(
    df: pd.DataFrame,
    target: str = "los_days",
    include_suit_features: bool = True,
    scale: bool = False,
    judge_target_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for tree-based models (Random Forest, XGBoost, Decision Tree).

    Features included:
      - Complexity metrics (n_events, activity_entropy, etc.)
      - Case attributes (plaintiffs, defendants, counts)
      - Party types
      - Suit structure (n_unique_suits, suit_entropy, has_<suit>, ...)
      - City (one-hot encoded)

    Features excluded:
      - case_type (raw categorical, collinear with nature_suit)
      - District_Judge (high cardinality)
      - nature_suit (raw, correlated with case_type, but features extracted)

    Args:
        df: Input DataFrame from aggregations/by_case.parquet
        target: Target column name (default: "los_days")
        include_suit_features: Whether to extract features from nature_suits array
        scale: Whether to StandardScale features (not needed for trees, but optional)

    Returns:
        Tuple of (X, y) where:
            X: DataFrame with features
            y: Series with target values
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    # Prepare target
    y = df[target].copy()

    # Remove rows with missing target
    valid_idx = y.notna() & (y >= 0)
    df = df[valid_idx].copy()
    y = y[valid_idx].copy()

    # 1. Numeric features
    numeric_cols = _get_numeric_feature_cols(df)
    X_numeric = _prepare_numeric_features(df, numeric_cols)

    # 2. Suit structure features (extracted from nature_suits array)
    X_suit = pd.DataFrame(index=df.index)
    if include_suit_features and "nature_suits" in df.columns:
        X_suit = _prepare_suit_features(df)

    # 3. Categorical features (city)
    X_categorical = _prepare_categorical_features(df, CATEGORICAL_COLS)

    # 4. Case-level flags (is_cv, is_mdl)
    X_flags = _prepare_case_flags(df)

    # 5. Judge feature: target encoding when map provided, label encoding otherwise
    X_judge = _prepare_judge_feature(df, judge_target_map)

    # Combine all features
    X = pd.concat([X_numeric, X_suit, X_categorical, X_flags, X_judge], axis=1)

    # Reset index for sklearn compatibility
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Optional: Scale (not needed for trees, but can help some models)
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y


def prepare_for_trees_judge_level(
    df: pd.DataFrame,
    target: str = "los_mean",
    scale: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare judge-level aggregated data for tree models.

    Args:
        df: Input DataFrame from aggregations/by_judge.parquet
        target: Target column (e.g., "los_mean", "los_median")
        scale: Whether to StandardScale features

    Returns:
        Tuple of (X, y)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    # Judge-level features (all numeric averages)
    exclude_judge_cols = ["District_Judge", "n_cases", "pct_cv", target]
    X = df.drop(columns=exclude_judge_cols, errors="ignore")

    y = df[target].copy()

    # Remove rows with NaN target
    valid_idx = y.notna()
    X = X[valid_idx].copy()
    y = y[valid_idx].copy()

    # Impute missing
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        X_imputed = pd.DataFrame(X_scaled, columns=X_imputed.columns)

    X_imputed = X_imputed.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X_imputed, y


def get_feature_names(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Get feature names by category for interpretability.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with feature categories and their column names
    """
    numeric_cols = _get_numeric_feature_cols(df)
    suit_cols = []
    if "nature_suits" in df.columns:
        suit_features_df = build_suit_features_dataframe(df["nature_suits"])
        suit_cols = get_suit_feature_names(suit_features_df)

    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    cat_encoded_cols = []
    if categorical_cols:
        X_cat = df[categorical_cols].copy()
        X_cat_encoded = pd.get_dummies(X_cat, columns=categorical_cols, drop_first=True)
        cat_encoded_cols = X_cat_encoded.columns.tolist()

    return {
        "complexity": [c for c in COMPLEXITY_CORE if c in numeric_cols],
        "case_attributes": [c for c in CASE_ATTRIBUTES if c in numeric_cols],
        "party_types": [c for c in PARTY_TYPES if c in numeric_cols],
        "event_attributes": [c for c in KEY_ATTRIBUTES if c in numeric_cols],
        "suit_structure": suit_cols,
        "categorical": cat_encoded_cols,
    }


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    df = pd.read_parquet(Path(__file__).parent.parent / "data" / "aggregations" / "by_case.parquet")

    print("=== Preparing data for trees ===\n")
    X, y = prepare_for_trees(df)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nFeature columns ({len(X.columns)}):")
    print(X.columns.tolist())

    print(f"\nTarget statistics:")
    print(f"  Mean: {y.mean():.1f} days")
    print(f"  Median: {y.median():.1f} days")
    print(f"  Std: {y.std():.1f} days")
    print(f"  Min: {y.min():.1f} days")
    print(f"  Max: {y.max():.1f} days")

    feature_info = get_feature_names(df)
    print(f"\nFeature breakdown:")
    for category, cols in feature_info.items():
        print(f"  {category}: {len(cols)} features")
