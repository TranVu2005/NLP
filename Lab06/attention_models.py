"""PyTorch models + data/training helpers for LAB 06 (Experiment 4, Application, error analysis).

The self-attention layer here is hand-written (Q/K/V linear maps, manual
softmax, A @ V); torch.nn.MultiheadAttention is NOT used. The data pipeline
re-uses LAB 05 (same CSVs, same tokenizer, same vocabulary, same split).
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS

HERE = Path(__file__).resolve().parent
LAB05 = HERE.parent / "Lab05"
if str(LAB05) not in sys.path:
    sys.path.insert(0, str(LAB05))

import sequence_models as lab05  # noqa: E402  (LAB 05 pipeline: tokenizer, Vocabulary, split)

PAD, UNK = lab05.PAD_INDEX, lab05.UNK_INDEX
MAX_LEN, EMB_DIM, BATCH, LR, EPOCHS = 100, 64, 128, 1e-3, 3  # identical to LAB 05 base config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data
@dataclass
class Data:
    vocab: Any
    train: Any  # DataFrames with columns text,label,sample_id
    val: Any
    test: Any
    source: str
    report: dict = field(default_factory=dict)


def load_lab05_data() -> Data:
    """Load LAB 05 CSVs and rebuild its leakage-safe train/val/test split + vocabulary."""
    train_raw, test_raw, source = lab05.load_imdb_dataset()
    train, val, test, report = lab05.make_splits(train_raw, test_raw, seed=lab05.SEED)
    vocab = lab05.Vocabulary.from_texts(train["text"], max_size=lab05.MAX_VOCAB_SIZE)
    return Data(vocab, train, val, test, source, report)


def make_dataset(frame: Any, vocab: Any) -> Any:
    """Encode a DataFrame with the LAB 05 tokenizer; truncate/pad to MAX_LEN (keep first tokens)."""
    seqs = [lab05.encode_text(t, vocab) for t in frame["text"]]
    return lab05.ReviewDataset(seqs, frame["label"].tolist(), MAX_LEN)


def make_loader(dataset: Any, shuffle: bool, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, generator=gen if shuffle else None)


def encode_batch(texts: Sequence[str], vocab: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    """Texts -> (ids (B, MAX_LEN), lengths (B,)) using the LAB 05 tokenizer/truncation."""
    rows = [lab05.pad_sequence(lab05.encode_text(t, vocab), MAX_LEN) for t in texts]
    ids = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    lengths = torch.tensor([max(r[1], 1) for r in rows], dtype=torch.long, device=device)
    return ids, lengths


# --------------------------------------------------------------------------- models
class LSTMClassifier(nn.Module):
    """Embedding -> LSTM -> final hidden state (last valid step) -> linear logit."""

    def __init__(self, vocab_size: int, emb: int = EMB_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, ids: Tensor, lengths: Tensor) -> Tensor:
        x = self.embedding(ids)  # (B, T, emb)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (1, B, hidden) = h at last valid token
        return self.classifier(h_n[-1]).squeeze(-1)  # (B,)


class SelfAttentionLayer(nn.Module):
    """Single-head scaled dot-product self-attention, written by hand.

    Q = X W_Q, K = X W_K, V = X W_V ;  A = softmax(Q K^T / sqrt(d_k)) ;  C = A V
    """

    def __init__(self, d_model: int, d_k: int | None = None) -> None:
        super().__init__()
        d_k = d_k or d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, X: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor]:
        """X (B, T, d_model); key_mask (B, T) bool, True = real token. Returns C (B,T,d), A (B,T,T)."""
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)  # (B, T, d_k)
        S = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)  # (B, T, T)
        S = S.masked_fill(~key_mask[:, None, :], -1e30)  # padded keys get ~0 weight
        S = S - S.max(dim=-1, keepdim=True).values  # stable softmax (subtract row max)
        E = torch.exp(S)
        A = E / E.sum(dim=-1, keepdim=True)  # (B, T, T), rows sum to 1
        return A @ V, A  # context (B, T, d_k)


class AttentionClassifier(nn.Module):
    """Embedding + learnable position emb -> self-attention -> masked mean pooling -> linear logit.

    Self-attention alone is permutation-equivariant, so a learnable position
    embedding (max 100 positions) is added to inject word order.
    """

    def __init__(self, vocab_size: int, emb: int = EMB_DIM, max_len: int = MAX_LEN) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, emb)
        self.attn = SelfAttentionLayer(emb)
        self.classifier = nn.Linear(emb, 1)

    def forward(self, ids: Tensor, lengths: Tensor, return_attention: bool = False):
        B, T = ids.shape
        key_mask = torch.arange(T, device=ids.device)[None, :] < lengths[:, None]  # (B, T)
        X = self.embedding(ids) + self.pos(torch.arange(T, device=ids.device))[None]  # (B, T, emb)
        C, A = self.attn(X, key_mask)
        m = key_mask.unsqueeze(-1).to(C.dtype)
        pooled = (C * m).sum(1) / m.sum(1).clamp(min=1)  # mean over valid positions: (B, emb)
        logit = self.classifier(pooled).squeeze(-1)
        return (logit, A) if return_attention else logit


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for ids, lengths, y in loader:
        ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
        logit = model(ids, lengths)
        loss_sum += bce(logit, y).item()
        n += len(y)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int), loss_sum / n


def metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict[str, float]:
    pred = (p_pos >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "f1_macro": f1_score(y_true, pred, average="macro")}


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                epochs: int = EPOCHS, log: bool = True) -> dict[str, Any]:
    """Adam(lr=1e-3), BCE-with-logits, fixed epochs. Training time excludes validation passes."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    train_time, history = 0.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        total, count = 0.0, 0
        for ids, lengths, y in train_loader:
            ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = bce(model(ids, lengths), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        p, y_val, v_loss = predict_loader(model, val_loader, device)
        acc = metrics(y_val, p)["accuracy"]
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": v_loss, "val_acc": acc})
        if log:
            print(f"  epoch {epoch}: train_loss={total / count:.4f} val_loss={v_loss:.4f} val_acc={acc:.4f}")
    return {"train_time": train_time, "history": history}


def confusion(y_true: np.ndarray, p_pos: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, (p_pos >= 0.5).astype(int), labels=[0, 1])


# --------------------------------------------------------------------------- inference helpers
@torch.no_grad()
def explain(model: AttentionClassifier, text: str, vocab: Any, device: torch.device) -> dict[str, Any]:
    """Run one text; return tokens (after LAB 05 truncation), p(positive), A (L, L) and received attention."""
    model.eval()
    tokens = lab05.tokenize_text(text)[:MAX_LEN]
    ids, lengths = encode_batch([text], vocab, device)
    logit, A = model(ids, lengths, return_attention=True)
    L = int(lengths.item()) if tokens else 1
    A = A[0, :L, :L].cpu().numpy()
    return {"tokens": tokens, "p_pos": float(torch.sigmoid(logit).item()), "A": A,
            "received": A.mean(axis=0)}  # mean over queries of attention each token receives


@torch.no_grad()
def prob_positive(model: nn.Module, token_lists: Sequence[Sequence[str]], vocab: Any, device: torch.device) -> np.ndarray:
    """p(positive) for already-tokenised inputs (used for deletion/perturbation tests)."""
    model.eval()
    rows = []
    for toks in token_lists:
        ids = [vocab.token_to_id.get(t, UNK) for t in toks][:MAX_LEN] or [UNK]
        rows.append(lab05.pad_sequence(ids, MAX_LEN))
    ids = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    lengths = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    return torch.sigmoid(model(ids, lengths)).cpu().numpy()
