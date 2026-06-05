#!/usr/bin/env python3
"""
Build and manage judge vocabulary for embeddings.

Converts judge IDs to integer indices for embedding layer.
Handles missing judges as "Unknown" category.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class JudgeVocabulary:
    """Build and store judge vocabulary for embeddings."""

    def __init__(self):
        self.judge_to_id = {}
        self.id_to_judge = {}
        self.vocab_size = 0

    def build(self, judge_series: pd.Series):
        """
        Build vocabulary from judge IDs.

        Args:
            judge_series: pd.Series with judge names/IDs
        """
        # Get unique judges (excluding NaN)
        unique_judges = sorted(judge_series.dropna().unique())

        # Special tokens
        special_tokens = ["<UNK>", "<UNKNOWN>"]
        self.judge_to_id = {token: i for i, token in enumerate(special_tokens)}

        # Add judges
        for i, judge in enumerate(unique_judges, start=len(special_tokens)):
            self.judge_to_id[judge] = i

        self.id_to_judge = {v: k for k, v in self.judge_to_id.items()}
        self.vocab_size = len(self.judge_to_id)

        print(f"Judge vocabulary built:")
        print(f"  Total judges: {len(unique_judges)}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Sample judges: {unique_judges[:5]}")

    def encode(self, judge_id: str | None) -> int:
        """
        Convert judge ID to integer.

        Args:
            judge_id: Judge identifier or None (missing)

        Returns:
            Integer index for embedding
        """
        if judge_id is None or pd.isna(judge_id):
            return self.judge_to_id["<UNK>"]

        judge_str = str(judge_id).strip()
        if judge_str == "" or judge_str.lower() in ["none", "nan"]:
            return self.judge_to_id["<UNK>"]

        return self.judge_to_id.get(judge_str, self.judge_to_id["<UNK>"])

    def encode_series(self, judge_series: pd.Series) -> list[int]:
        """
        Encode entire series of judge IDs.

        Args:
            judge_series: pd.Series with judge IDs

        Returns:
            List of integer indices
        """
        return [self.encode(j) for j in judge_series]

    def save(self, path: Path):
        """Save vocabulary to JSON."""
        with open(path, "w") as f:
            json.dump(self.judge_to_id, f, indent=2)
        print(f"Judge vocabulary saved to {path}")

    def load(self, path: Path):
        """Load vocabulary from JSON."""
        with open(path, "r") as f:
            self.judge_to_id = json.load(f)
        self.id_to_judge = {int(v): k for k, v in self.judge_to_id.items()}
        self.vocab_size = len(self.judge_to_id)
        print(f"Judge vocabulary loaded from {path}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    df = pd.read_parquet(Path(__file__).parent.parent / "data" / "aggregations" / "by_case.parquet")

    print("=== Judge Vocabulary Example ===\n")
    vocab = JudgeVocabulary()
    vocab.build(df["District_Judge"])

    print(f"\nEncoding examples:")
    test_judges = [
        df["District_Judge"].iloc[0],
        df["District_Judge"].iloc[100],
        None,
        "InvalidJudge"
    ]
    for j in test_judges:
        encoded = vocab.encode(j)
        print(f"  Judge {j:15s} → ID {encoded}")
