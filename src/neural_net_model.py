#!/usr/bin/env python3
"""
PyTorch Neural Network for predicting LOS with embeddings.

Architecture:
  - Numeric features input (scaled)
  - Judge embedding layer (learns judge-specific effects)
  - Suit embedding layer (learns suit type patterns)
  - Dense hidden layers combining all features
  - Output: LOS prediction

This model captures:
  1. Case complexity patterns (through numeric features)
  2. Judge specialization/speed (through embeddings)
  3. Suit composition diversity (through embeddings)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CourtCaseNeuralNet(nn.Module):
    """
    Neural network for LOS prediction with judge and suit embeddings.

    Inputs:
      - numeric_features: (batch_size, n_numeric_features)
      - judges_encoded: (batch_size,) - judge IDs
      - suits_encoded: (batch_size, max_suit_length) - suit ID sequences
      - suits_mask: (batch_size, max_suit_length) - padding mask (1=real, 0=pad)
    """

    def __init__(
        self,
        n_numeric_features: int,
        judge_vocab_size: int,
        suit_vocab_size: int | None = None,
        judge_embedding_dim: int = 16,
        suit_embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
        use_suit_embeddings: bool = True,
    ):
        """
        Initialize Neural Network.

        Args:
            n_numeric_features: Number of numeric input features
            judge_vocab_size: Size of judge vocabulary
            suit_vocab_size: Size of suit vocabulary (if using suit embeddings)
            judge_embedding_dim: Dimension of judge embeddings
            suit_embedding_dim: Dimension of suit embeddings
            hidden_dim: Size of hidden layers
            dropout_rate: Dropout probability
            use_suit_embeddings: Whether to include suit embeddings
        """
        super().__init__()

        self.n_numeric_features = n_numeric_features
        self.use_suit_embeddings = use_suit_embeddings

        # Judge embedding layer
        self.judge_embedding = nn.Embedding(
            num_embeddings=judge_vocab_size,
            embedding_dim=judge_embedding_dim,
            padding_idx=0  # <UNK> or <PAD> token
        )

        # Suit embedding layer (optional)
        if use_suit_embeddings and suit_vocab_size is not None:
            self.suit_embedding = nn.Embedding(
                num_embeddings=suit_vocab_size,
                embedding_dim=suit_embedding_dim,
                padding_idx=0  # <PAD> token
            )
        else:
            self.suit_embedding = None
            suit_embedding_dim = 0

        # Calculate combined input size
        combined_size = (
            n_numeric_features
            + judge_embedding_dim
            + suit_embedding_dim
        )

        # Dense layers
        self.fc1 = nn.Linear(combined_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.output = nn.Linear(hidden_dim // 4, 1)

    def forward(
        self,
        numeric_features: torch.Tensor,
        judges_encoded: torch.Tensor,
        suits_encoded: torch.Tensor | None = None,
        suits_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            numeric_features: (batch_size, n_numeric_features)
            judges_encoded: (batch_size,) - judge IDs
            suits_encoded: (batch_size, max_suit_length) - suit sequences
            suits_mask: (batch_size, max_suit_length) - padding mask

        Returns:
            (batch_size,) - LOS predictions
        """
        batch_size = numeric_features.shape[0]

        # Numeric features (already scaled)
        x_numeric = numeric_features

        # Judge embeddings
        x_judge = self.judge_embedding(judges_encoded)  # (batch_size, judge_embedding_dim)

        # Suit embeddings (if using)
        x_suit = None
        if self.use_suit_embeddings and self.suit_embedding is not None and suits_encoded is not None:
            suit_embeddings = self.suit_embedding(suits_encoded)  # (batch_size, max_suit_length, suit_embedding_dim)

            # Average pooling with masking (ignore padding)
            if suits_mask is not None:
                suits_mask_expanded = suits_mask.unsqueeze(-1).float()  # (batch_size, max_suit_length, 1)
                suit_embeddings_masked = suit_embeddings * suits_mask_expanded
                x_suit = suit_embeddings_masked.sum(dim=1) / suits_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            else:
                x_suit = suit_embeddings.mean(dim=1)
            # Result: (batch_size, suit_embedding_dim)

        # Combine all inputs
        if x_suit is not None:
            x = torch.cat([x_numeric, x_judge, x_suit], dim=1)
        else:
            x = torch.cat([x_numeric, x_judge], dim=1)

        # Forward through dense layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Output
        output = self.output(x)
        return output.squeeze(-1)


class CourtCaseNeuralNetLSTM(nn.Module):
    """
    Neural network with LSTM for sequential suit processing.

    The LSTM captures temporal patterns in suit sequences
    (e.g., certain suit types appear before others).
    """

    def __init__(
        self,
        n_numeric_features: int,
        judge_vocab_size: int,
        suit_vocab_size: int,
        judge_embedding_dim: int = 16,
        suit_embedding_dim: int = 16,
        lstm_hidden_dim: int = 32,
        dense_hidden_dim: int = 64,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Embeddings
        self.judge_embedding = nn.Embedding(
            num_embeddings=judge_vocab_size,
            embedding_dim=judge_embedding_dim,
            padding_idx=0
        )

        self.suit_embedding = nn.Embedding(
            num_embeddings=suit_vocab_size,
            embedding_dim=suit_embedding_dim,
            padding_idx=0
        )

        # LSTM for suit sequences
        self.lstm = nn.LSTM(
            input_size=suit_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if lstm_hidden_dim > 1 else 0
        )

        # Dense layers
        combined_size = n_numeric_features + judge_embedding_dim + lstm_hidden_dim
        self.fc1 = nn.Linear(combined_size, dense_hidden_dim)
        self.bn1 = nn.BatchNorm1d(dense_hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(dense_hidden_dim, dense_hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(dense_hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(dense_hidden_dim // 2, 1)

    def forward(
        self,
        numeric_features: torch.Tensor,
        judges_encoded: torch.Tensor,
        suits_encoded: torch.Tensor,
        suits_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with LSTM suit processing."""
        # Embed suits
        suit_embeddings = self.suit_embedding(suits_encoded)  # (batch_size, max_suit_length, suit_embedding_dim)

        # Get suit sequence lengths (non-padded count)
        suit_lengths = suits_mask.sum(dim=1).cpu()

        # Pack padded sequences
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            suit_embeddings,
            suit_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM
        _, (hidden, _) = self.lstm(packed)
        suit_features = hidden[-1]  # (batch_size, lstm_hidden_dim)

        # Judge embeddings
        judge_features = self.judge_embedding(judges_encoded)  # (batch_size, judge_embedding_dim)

        # Combine all
        x = torch.cat([numeric_features, judge_features, suit_features], dim=1)

        # Dense layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        output = self.output(x)
        return output.squeeze(-1)


if __name__ == "__main__":
    # Example usage
    print("=== Neural Network Models ===\n")

    # Initialize model
    model = CourtCaseNeuralNet(
        n_numeric_features=35,
        judge_vocab_size=93,
        suit_vocab_size=50,
        judge_embedding_dim=16,
        suit_embedding_dim=16,
        hidden_dim=128,
        dropout_rate=0.3,
    )

    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    print(f"\n=== Test Forward Pass ===")
    batch_size = 32
    numeric = torch.randn(batch_size, 35)
    judges = torch.randint(0, 93, (batch_size,))
    suits = torch.randint(0, 50, (batch_size, 20))
    mask = torch.ones(batch_size, 20)
    mask[:, 10:] = 0  # Mask out padding

    output = model(numeric, judges, suits, mask)
    print(f"Input numeric: {numeric.shape}")
    print(f"Input judges: {judges.shape}")
    print(f"Input suits: {suits.shape}")
    print(f"Output: {output.shape}")
    print(f"Output range: [{output.min():.1f}, {output.max():.1f}]")
