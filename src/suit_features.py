#!/usr/bin/env python3
"""
Extract tree-friendly features from nature_suits array.

Converts nature_suits list: ["contract", "contract", "ip", "employment"]
Into structural features: n_unique_suits, suit_entropy, has_contract, has_ip, ...

This avoids collinearity with case_type while capturing suit composition.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


def _suit_entropy(suit_counts: Counter) -> float:
    """Calculate Shannon entropy of suit distribution."""
    total = sum(suit_counts.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in suit_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def extract_suit_features(nature_suits_list: Optional[list[str]]) -> dict:
    """
    Extract tree-friendly features from nature_suits array.

    Args:
        nature_suits_list: List of suit types for one case
                          e.g., ["contract", "contract", "ip", "employment"]

    Returns:
        Dictionary with:
            - n_unique_suits: Number of different suit types
            - suit_entropy: Diversity of suit distribution (0 = all same, high = diverse)
            - is_multisuit: Binary flag (1 if >1 unique suit, else 0)
            - suit_dominance: Max proportion of any single suit type
            - has_<suittype>: Binary presence flags
            - suit_freq_<suittype>: Proportion of each suit type
    """
    features = {}

    if not nature_suits_list or len(nature_suits_list) == 0:
        # No suits case
        features["n_unique_suits"] = 0
        features["suit_entropy"] = 0.0
        features["is_multisuit"] = 0
        features["suit_dominance"] = 0.0
        return features

    # Count suits
    suit_counts = Counter(nature_suits_list)
    unique_suits = list(suit_counts.keys())
    n_unique = len(unique_suits)
    total = len(nature_suits_list)

    # Basic counts
    features["n_unique_suits"] = n_unique
    features["is_multisuit"] = 1 if n_unique > 1 else 0

    # Entropy (diversity)
    features["suit_entropy"] = _suit_entropy(suit_counts)

    # Suit dominance (concentration)
    max_count = max(suit_counts.values())
    features["suit_dominance"] = max_count / total

    # Frequency of each suit (raw count normalized)
    for suit, count in suit_counts.items():
        # Store as suit_freq_<suitname> for tree feature names
        normalized_name = suit.lower().replace(" ", "_").replace("/", "_")
        features[f"suit_freq_{normalized_name}"] = count / total

    # Presence flags (binary: has_<suit>)
    for suit in unique_suits:
        normalized_name = suit.lower().replace(" ", "_").replace("/", "_")
        features[f"has_{normalized_name}"] = 1

    return features


def build_suit_features_dataframe(nature_suits_series: pd.Series) -> pd.DataFrame:
    """
    Extract suit features for all cases.

    Args:
        nature_suits_series: pd.Series with nature_suits array per case

    Returns:
        DataFrame with suit feature columns (one row per case)
    """
    if nature_suits_series is None or len(nature_suits_series) == 0:
        raise ValueError("nature_suits_series is empty")

    # Extract features for all cases
    suit_features_list = []
    for suits_array in nature_suits_series:
        features = extract_suit_features(suits_array)
        suit_features_list.append(features)

    # Convert to DataFrame
    suit_features_df = pd.DataFrame(suit_features_list)

    # Fill NaN with 0 (handles missing suit types)
    suit_features_df = suit_features_df.fillna(0.0)

    return suit_features_df


def get_suit_feature_names(suit_features_df: pd.DataFrame) -> list[str]:
    """Get sorted list of suit feature column names."""
    return sorted(suit_features_df.columns.tolist())


class SuitVocabulary:
    """Build and store suit vocabulary for embeddings."""

    def __init__(self):
        self.suit_to_id = {}
        self.id_to_suit = {}
        self.vocab_size = 0

    def build(self, nature_suits_lists: list[list[str] | None]):
        """
        Build vocabulary from all suits in dataset.

        Args:
            nature_suits_lists: List of suit arrays per case
        """
        unique_suits = set()
        for suits_list in nature_suits_lists:
            if suits_list:
                unique_suits.update(suits_list)

        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>"]
        self.suit_to_id = {token: i for i, token in enumerate(special_tokens)}

        # Add suits (sorted for reproducibility)
        for i, suit in enumerate(sorted(unique_suits), start=len(special_tokens)):
            self.suit_to_id[suit] = i

        self.id_to_suit = {v: k for k, v in self.suit_to_id.items()}
        self.vocab_size = len(self.suit_to_id)

        print(f"Suit vocabulary built:")
        print(f"  Total unique suits: {len(unique_suits)}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Suits: {sorted(unique_suits)}")

    def encode(self, suits_list: list[str] | None, max_length: int = 20) -> list[int]:
        """
        Convert suit list to integer IDs with padding.

        Args:
            suits_list: List of suit names (or None)
            max_length: Length to pad/truncate to

        Returns:
            List of integer IDs (padded to max_length)
        """
        if not suits_list:
            return [self.suit_to_id["<PAD>"]] * max_length

        encoded = [
            self.suit_to_id.get(s, self.suit_to_id["<UNK>"]) for s in suits_list[:max_length]
        ]
        # Pad to max_length
        encoded += [self.suit_to_id["<PAD>"]] * (max_length - len(encoded))
        return encoded

    def save(self, path: Path):
        """Save vocabulary to JSON."""
        import json
        with open(path, "w") as f:
            json.dump(self.suit_to_id, f, indent=2)
        print(f"Suit vocabulary saved to {path}")

    def load(self, path: Path):
        """Load vocabulary from JSON."""
        import json
        with open(path, "r") as f:
            self.suit_to_id = json.load(f)
        self.id_to_suit = {int(v): k for k, v in self.suit_to_id.items()}
        self.vocab_size = len(self.suit_to_id)
        print(f"Suit vocabulary loaded from {path}")


if __name__ == "__main__":
    # Example usage
    example_cases = [
        ["contract", "contract"],
        ["employment", "employment", "contract"],
        None,
        [],
        ["ip", "patent", "trademark"],
    ]

    print("Example suit feature extraction:\n")
    for i, suits in enumerate(example_cases):
        features = extract_suit_features(suits)
        print(f"Case {i}: {suits}")
        print(f"  → n_unique_suits: {features.get('n_unique_suits')}")
        print(f"  → suit_entropy: {features.get('suit_entropy', 0):.3f}")
        print(f"  → is_multisuit: {features.get('is_multisuit')}")
        print(f"  → suit_dominance: {features.get('suit_dominance', 0):.3f}")
        print()

    print("\nExample suit vocabulary:\n")
    vocab = SuitVocabulary()
    vocab.build(example_cases)
    print(f"Encoding first case: {example_cases[0]}")
    print(f"  → {vocab.encode(example_cases[0])}")
