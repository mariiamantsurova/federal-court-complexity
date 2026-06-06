#!/usr/bin/env python3
"""
Neural Network for LOS prediction.

Two judge-representation modes:
  default              — learned embedding (randomly initialised, trained end-to-end)
  --use-hf-embeddings  — pretrained sentence embeddings from
                         'sentence-transformers/all-MiniLM-L6-v2' (frozen),
                         projected to embedding_dim via a trainable linear layer

Architecture (both modes):
  numeric (scaled) ──────────────────────────────────────┐
  judge → embedding (16-dim) ────────────────────────────┼─ cat → FC→BN→ReLU→Drop
                                                          │       → FC→BN→ReLU→Drop
                                                          │       → FC→BN→ReLU→Drop
                                                          └──────→ FC(1) = LOS (days)

Note: nature_suits column is absent from by_case.parquet, so suit embeddings
      are disabled regardless of model choice.

Outputs:
  docs/04_neural_net_results{suffix}.json
  reports/figures/04_nn_loss_curve{suffix}.png

Usage:
  .venv/bin/python3 scripts/run_neural_net.py
  .venv/bin/python3 scripts/run_neural_net.py --use-hf-embeddings
  .venv/bin/python3 scripts/run_neural_net.py --case-type cv --epochs 30
  .venv/bin/python3 scripts/run_neural_net.py --use-hf-embeddings --case-type cv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing_neural_net import prepare_for_neural_net


# ── Model definitions ─────────────────────────────────────────────────────────

class CourtNNBase(nn.Module):
    """Dense head shared by both judge-embedding variants."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CourtNNLearnedEmbedding(nn.Module):
    """
    Custom model: judge embedding learned end-to-end.
    No pretrained components.
    """

    def __init__(
        self,
        n_numeric: int,
        judge_vocab_size: int,
        judge_embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.judge_emb = nn.Embedding(judge_vocab_size, judge_embedding_dim, padding_idx=0)
        self.head = CourtNNBase(n_numeric + judge_embedding_dim, hidden_dim, dropout)

    def forward(self, numeric: torch.Tensor, judge_ids: torch.Tensor) -> torch.Tensor:
        j = self.judge_emb(judge_ids)                  # (B, emb_dim)
        x = torch.cat([numeric, j], dim=1)
        return self.head(x)


class CourtNNHuggingFace(nn.Module):
    """
    HuggingFace variant: judge represented by frozen sentence-transformer
    embeddings (all-MiniLM-L6-v2, 384-dim), projected to judge_embedding_dim
    via a trainable linear layer.

    The judge_emb_matrix is a (vocab_size, 384) float tensor of precomputed
    sentence embeddings — passed in at construction, registered as a buffer
    (not a parameter, so not updated by optimiser).
    """

    def __init__(
        self,
        n_numeric: int,
        judge_emb_matrix: torch.Tensor,  # (vocab_size, hf_dim)
        judge_embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        hf_dim = judge_emb_matrix.shape[1]
        self.register_buffer("judge_emb_matrix", judge_emb_matrix)  # frozen
        self.proj = nn.Linear(hf_dim, judge_embedding_dim)           # trainable
        self.head = CourtNNBase(n_numeric + judge_embedding_dim, hidden_dim, dropout)

    def forward(self, numeric: torch.Tensor, judge_ids: torch.Tensor) -> torch.Tensor:
        raw = self.judge_emb_matrix[judge_ids]   # (B, hf_dim) — lookup, no grad
        j   = self.proj(raw)                     # (B, judge_embedding_dim)
        x   = torch.cat([numeric, j], dim=1)
        return self.head(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(case_type: str | None, exclude_mdl: bool) -> pd.DataFrame:
    path = ROOT / "data" / "aggregations" / "by_case.parquet"
    df = pd.read_parquet(path)
    if case_type in ("cv", "cr"):
        df = df[df["case_type"] == case_type]
    if exclude_mdl and "is_mdl" in df.columns:
        df = df[~df["is_mdl"]]
    print(f"Loaded {len(df):,} cases  (case_type={case_type or 'all'}, exclude_mdl={exclude_mdl})")
    return df


def build_suffix(case_type: str | None, exclude_mdl: bool, hf: bool) -> str:
    parts = []
    if case_type:
        parts.append(case_type)
    if exclude_mdl:
        parts.append("no_mdl")
    if hf:
        parts.append("hf")
    return ("_" + "_".join(parts)) if parts else ""


def encode_judge_names_with_hf(
    judge_vocab,   # JudgeVocabulary instance
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> torch.Tensor:
    """
    Encode every judge name in the vocabulary as a sentence embedding.
    Returns a (vocab_size, 384) float32 tensor.
    Judges are encoded as 'Judge <name>' to give the model useful context.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed. Run: .venv/bin/pip install sentence-transformers")

    print(f"Loading HuggingFace model: {model_name}")
    st_model = SentenceTransformer(model_name)

    id_to_judge = judge_vocab.id_to_judge           # {int_id: name}
    vocab_size   = judge_vocab.vocab_size
    hf_dim       = 384                              # all-MiniLM-L6-v2 output dim

    sentences = []
    for idx in range(vocab_size):
        name = id_to_judge.get(idx, "<UNK>")
        # Special tokens stay as-is; real names get context prefix
        if name.startswith("<"):
            sentences.append(name)
        else:
            sentences.append(f"Judge {name}")

    print(f"  Encoding {len(sentences)} judge names ...")
    embeddings = st_model.encode(sentences, batch_size=64, show_progress_bar=False)
    return torch.tensor(embeddings, dtype=torch.float32)


def make_tensors(prepared_data: dict, y: pd.Series):
    numeric = torch.tensor(prepared_data["numeric_features"], dtype=torch.float32)
    judges  = torch.tensor(prepared_data["judges_encoded"],   dtype=torch.long)
    target  = torch.tensor(y.values, dtype=torch.float32)
    return numeric, judges, target


def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for numeric, judges, target in loader:
        numeric, judges, target = numeric.to(device), judges.to(device), target.to(device)
        optimiser.zero_grad()
        pred = model(numeric, judges)
        loss = criterion(pred, target)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(target)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for numeric, judges, target in loader:
        numeric, judges, target = numeric.to(device), judges.to(device), target.to(device)
        pred = model(numeric, judges)
        total_loss += criterion(pred, target).item() * len(target)
        preds.append(pred.cpu())
        targets.append(target.cpu())
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(loader.dataset), preds, targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None)
    parser.add_argument("--exclude-mdl", action="store_true")
    parser.add_argument("--use-hf-embeddings", action="store_true",
                        help="Use sentence-transformers for judge embeddings (frozen)")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--batch-size",    type=int,   default=1024)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--hidden-dim",    type=int,   default=128)
    parser.add_argument("--embedding-dim", type=int,   default=16)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--patience",      type=int,   default=8,
                        help="Early stopping patience (val loss epochs)")
    args = parser.parse_args()

    suffix   = build_suffix(args.case_type, args.exclude_mdl, args.use_hf_embeddings)
    docs_dir = ROOT / "docs"
    figs_dir = ROOT / "reports" / "figures"
    docs_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    df = load_data(args.case_type, args.exclude_mdl)
    prepared_data, y = prepare_for_neural_net(df)

    n_numeric     = prepared_data["numeric_features"].shape[1]
    judge_vocab   = prepared_data["judge_vocab"]
    judge_vocab_size = prepared_data["judges_vocab_size"]
    print(f"Numeric features: {n_numeric}  |  Judge vocab: {judge_vocab_size}")

    numeric, judges, target = make_tensors(prepared_data, y)

    # train/val/test split: 70 / 15 / 15
    idx = np.arange(len(target))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.15, random_state=42)
    idx_train, idx_val     = train_test_split(idx_trainval, test_size=0.15 / 0.85, random_state=42)

    def split(t):
        return t[idx_train], t[idx_val], t[idx_test]

    num_tr, num_val, num_te = split(numeric)
    jud_tr, jud_val, jud_te = split(judges)
    tgt_tr, tgt_val, tgt_te = split(target)

    train_ds = TensorDataset(num_tr, jud_tr, tgt_tr)
    val_ds   = TensorDataset(num_val, jud_val, tgt_val)
    test_ds  = TensorDataset(num_te, jud_te, tgt_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    if args.use_hf_embeddings:
        print("\n── HuggingFace judge embeddings (sentence-transformers/all-MiniLM-L6-v2) ──")
        hf_matrix = encode_judge_names_with_hf(judge_vocab)
        model = CourtNNHuggingFace(
            n_numeric         = n_numeric,
            judge_emb_matrix  = hf_matrix,
            judge_embedding_dim = args.embedding_dim,
            hidden_dim        = args.hidden_dim,
            dropout           = args.dropout,
        )
        mode_label = "HuggingFace (all-MiniLM-L6-v2 + projection)"
    else:
        print("\n── Custom learned judge embeddings ──")
        model = CourtNNLearnedEmbedding(
            n_numeric         = n_numeric,
            judge_vocab_size  = judge_vocab_size,
            judge_embedding_dim = args.embedding_dim,
            hidden_dim        = args.hidden_dim,
            dropout           = args.dropout,
        )
        mode_label = "Custom (learned embedding)"

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=4
    )

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}) ...")
    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    t_start        = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss          = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, _, _   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss - 1.0  # 1-day² improvement threshold
        if improved:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            rmse_tr = tr_loss ** 0.5
            rmse_va = val_loss ** 0.5
            print(f"  ep {epoch:3d}  train RMSE={rmse_tr:.1f}  val RMSE={rmse_va:.1f}"
                  + ("  *" if improved else ""))

        if no_improve >= args.patience:
            print(f"  Early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"Training complete in {elapsed:.1f}s")

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── test evaluation ───────────────────────────────────────────────────────
    _, y_pred_arr, y_true_arr = eval_epoch(model, test_loader, criterion, device)
    mae  = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = mean_squared_error(y_true_arr, y_pred_arr) ** 0.5
    r2   = r2_score(y_true_arr, y_pred_arr)
    print(f"\nTest metrics:  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

    # ── loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, [l ** 0.5 for l in train_losses], label="Train RMSE")
    ax.plot(epochs_range, [l ** 0.5 for l in val_losses],   label="Val RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (days)")
    ct_label = {"cv": "Civil", "cr": "Criminal"}.get(args.case_type or "", "All")
    ax.set_title(f"NN Loss Curve  [{ct_label}]  ({mode_label})")
    ax.legend()
    fig.tight_layout()
    loss_path = figs_dir / f"04_nn_loss_curve{suffix}.png"
    fig.savefig(loss_path, dpi=120)
    plt.close(fig)
    print(f"Saved {loss_path.relative_to(ROOT)}")

    # ── save model ────────────────────────────────────────────────────────────
    model_path = ROOT / "docs" / f"04_nn_model{suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path.relative_to(ROOT)}")

    # ── results JSON ──────────────────────────────────────────────────────────
    results = {
        "model": "NeuralNet",
        "judge_mode": "hf_sentence_transformer" if args.use_hf_embeddings else "learned_embedding",
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2" if args.use_hf_embeddings else None,
        "case_type": args.case_type or "all",
        "exclude_mdl": args.exclude_mdl,
        "n_train": int(len(idx_train)),
        "n_val":   int(len(idx_val)),
        "n_test":  int(len(idx_test)),
        "n_numeric_features": int(n_numeric),
        "judge_vocab_size": int(judge_vocab_size),
        "hyperparams": {
            "epochs_run": len(train_losses),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        },
        "metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)},
        "best_val_rmse": round(best_val_loss ** 0.5, 2),
        "train_time_s": round(elapsed, 1),
    }
    json_path = docs_dir / f"04_neural_net_results{suffix}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {json_path.relative_to(ROOT)}")

    print(f"\nDone.  MAE={mae:.1f} days  R²={r2:.4f}  [{mode_label}]")


if __name__ == "__main__":
    main()
