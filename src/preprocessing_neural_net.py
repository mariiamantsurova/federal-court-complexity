#!/usr/bin/env python3
"""
Preprocessing for Neural Network models with embeddings.

Features:
  - Numeric features (scaled): complexity, case attributes, party counts
  - Suit embeddings: array of suit IDs per case (learned embeddings)
  - Judge embeddings: single judge ID per case (learned embeddings)

Different from trees:
  - Includes District_Judge (as embedding)
  - Includes nature_suits array (as embedding)
  - All numeric features StandardScaled
  - No one-hot encoding (uses embeddings instead)

Usage:
  prepared_data, y = prepare_for_neural_net(df)
  # Access: prepared_data['numeric_features'], prepared_data['suits_encoded'],
  #         prepared_data['judges_encoded'], etc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.judge_vocabulary import JudgeVocabulary
from src.suit_features import build_suit_features_dataframe


# Numeric features (same as trees, but these will be scaled)
COMPLEXITY_CORE = [
    "n_events",
    "n_activity_types",
    "n_motions",
    "activity_entropy",
    "complexity_index",
]

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
]

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

# Categorical features to embed (not one-hot)
EMBEDDING_FEATURES = {
    "judge": "District_Judge",  # Will be embedded
    "magistrate": "Magistrate_Judge",  # Could be embedded or kept numeric
}

# Other numeric (can keep as-is)
OTHER_NUMERIC = [
    "Magistrate_Judge",  # Binary, can keep numeric
]

TARGET = "los_days"


def _get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns for NN."""
    candidates = COMPLEXITY_CORE + CASE_ATTRIBUTES + PARTY_TYPES + OTHER_NUMERIC
    return [col for col in candidates if col in df.columns]


def _prepare_judge_embeddings(df: pd.DataFrame) -> dict:
    """
    Prepare judge embeddings.

    Args:
        df: Input DataFrame with District_Judge column

    Returns:
        Dictionary with:
            - judges_encoded: (n_cases,) integer array of judge IDs
            - judge_vocab: JudgeVocabulary instance
            - vocab_size: Size of judge vocabulary
    """
    if "District_Judge" not in df.columns:
        raise ValueError("District_Judge column not found")

    # Build vocabulary
    judge_vocab = JudgeVocabulary()
    judge_vocab.build(df["District_Judge"])

    # Encode judges
    judges_encoded = np.array(judge_vocab.encode_series(df["District_Judge"]))

    return {
        "judges_encoded": judges_encoded,  # (n_cases,)
        "judge_vocab": judge_vocab,
        "vocab_size": judge_vocab.vocab_size,
    }


def _prepare_suit_embeddings(df: pd.DataFrame, max_suit_length: int = 20) -> dict:
    """
    Prepare suit embeddings (same as NN preprocessing would).

    Args:
        df: Input DataFrame with nature_suits column
        max_suit_length: Maximum number of suits per case

    Returns:
        Dictionary with suit encoding info
    """
    if "nature_suits" not in df.columns:
        print("WARNING: nature_suits not in DataFrame, suit embeddings skipped")
        return {
            "suits_encoded": None,
            "suits_mask": None,
            "suit_vocab": None,
        }

    from src.suit_features import SuitVocabulary

    # Build vocabulary
    suit_vocab = SuitVocabulary()
    suit_vocab.build(df["nature_suits"].tolist())

    # Encode all cases
    suits_encoded = []
    suits_mask = []

    for suits_list in df["nature_suits"]:
        encoded = suit_vocab.encode(suits_list, max_length=max_suit_length)
        suits_encoded.append(encoded)

        # Create mask (1 = real suit, 0 = padding)
        if suits_list:
            mask = [1] * min(len(suits_list), max_suit_length)
            mask += [0] * (max_suit_length - len(mask))
        else:
            mask = [0] * max_suit_length
        suits_mask.append(mask)

    return {
        "suits_encoded": np.array(suits_encoded),  # (n_cases, max_suit_length)
        "suits_mask": np.array(suits_mask),        # (n_cases, max_suit_length)
        "suit_vocab": suit_vocab,
    }


def prepare_for_neural_net(
    df: pd.DataFrame,
    target: str = TARGET,
    max_suit_length: int = 20,
    scale_numeric: bool = True,
) -> tuple[dict, pd.Series]:
    """
    Prepare data for Neural Network with embeddings.

    Features:
      - Numeric (scaled): complexity, case attributes, party counts
      - Judge embedding: District_Judge → ID
      - Suit embedding: nature_suits array → ID array

    Args:
        df: Input DataFrame from aggregations/by_case.parquet
        target: Target column name
        max_suit_length: Max suits per case (padding/truncation)
        scale_numeric: Whether to StandardScale numeric features

    Returns:
        Tuple of (prepared_data, y) where prepared_data is dict with:
            - numeric_features: (n_cases, n_numeric_features) scaled
            - judges_encoded: (n_cases,) judge IDs
            - judges_vocab_size: Size of judge vocabulary
            - suits_encoded: (n_cases, max_suit_length) suit IDs or None
            - suits_mask: (n_cases, max_suit_length) padding mask or None
            - feature_names: List of numeric feature names
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    # Prepare target
    y = df[target].copy()

    # Remove rows with invalid target
    valid_idx = y.notna() & (y >= 0)
    df = df[valid_idx].copy()
    y = y[valid_idx].copy()

    # 1. Numeric features (scaled)
    numeric_cols = _get_numeric_feature_cols(df)
    X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())

    if scale_numeric:
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        X_numeric_scaled = np.asarray(X_numeric_scaled, dtype=np.float32)
    else:
        X_numeric_scaled = X_numeric.values.astype(np.float32)

    # 2. Judge embeddings
    judge_data = _prepare_judge_embeddings(df)

    # 3. Suit embeddings (if available)
    suit_data = _prepare_suit_embeddings(df, max_suit_length)

    # Prepare output
    prepared_data = {
        "numeric_features": X_numeric_scaled,           # (n_cases, n_numeric)
        "numeric_feature_names": numeric_cols,
        "judges_encoded": judge_data["judges_encoded"],  # (n_cases,)
        "judges_vocab_size": judge_data["vocab_size"],
        "judge_vocab": judge_data["judge_vocab"],
        "suits_encoded": suit_data["suits_encoded"],     # (n_cases, max_suit_length) or None
        "suits_mask": suit_data["suits_mask"],           # (n_cases, max_suit_length) or None
        "suit_vocab": suit_data["suit_vocab"],
    }

    # Reset indices for consistency
    y = y.reset_index(drop=True)

    return prepared_data, y


def prepare_for_neural_net_judge_level(
    df: pd.DataFrame,
    target: str = "los_mean",
) -> tuple[dict, pd.Series]:
    """
    Prepare judge-level data for Neural Network.

    Note: Judge-level data already aggregated, no per-judge embeddings.
    Just scale numeric features.

    Args:
        df: Input DataFrame from aggregations/by_judge.parquet
        target: Target column (e.g., "los_mean")

    Returns:
        Tuple of (prepared_data, y)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    # Judge-level features (all numeric)
    exclude_cols = ["District_Judge", "n_cases", "pct_cv", target]
    X = df.drop(columns=exclude_cols, errors="ignore")

    y = df[target].copy()

    # Remove invalid
    valid_idx = y.notna()
    X = X[valid_idx].copy()
    y = y[valid_idx].copy()

    # Impute and scale
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    prepared_data = {
        "numeric_features": X_scaled,
        "numeric_feature_names": X.columns.tolist(),
        "judges_encoded": None,  # No judge embeddings at judge level
        "judges_vocab_size": None,
    }

    y = y.reset_index(drop=True)

    return prepared_data, y


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    df = pd.read_parquet(Path(__file__).parent.parent / "data" / "aggregations" / "by_case.parquet")

    print("=== Neural Network Preprocessing Example ===\n")
    prepared_data, y = prepare_for_neural_net(df)

    print(f"Numeric features: {prepared_data['numeric_features'].shape}")
    print(f"  Columns: {prepared_data['numeric_feature_names']}")
    print(f"\nJudge embeddings:")
    print(f"  Vocab size: {prepared_data['judges_vocab_size']}")
    print(f"  Encoded shape: {prepared_data['judges_encoded'].shape}")
    print(f"  Sample IDs: {prepared_data['judges_encoded'][:10]}")

    if prepared_data['suits_encoded'] is not None:
        print(f"\nSuit embeddings:")
        print(f"  Encoded shape: {prepared_data['suits_encoded'].shape}")
        print(f"  Mask shape: {prepared_data['suits_mask'].shape}")
        print(f"  Sample suits: {prepared_data['suits_encoded'][0]}")
    else:
        print(f"\nSuit embeddings: NOT AVAILABLE (nature_suits column missing)")

    print(f"\nTarget (y): {y.shape}")
    print(f"  Mean: {y.mean():.1f} days")
    print(f"  Median: {y.median():.1f} days")
