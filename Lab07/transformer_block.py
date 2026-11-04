"""Transformer Encoder block tu ghep (LAB 07, muc 17-18) + classifier.

Core KHONG dung nn.TransformerEncoderLayer / nn.MultiheadAttention.
Chi dung primitive: nn.Linear, nn.LayerNorm, nn.Dropout, softmax.

Quy uoc mask: `pad_mask` la bool (B, T), True = vi tri PAD (khong duoc attend toi).
Block la post-norm: x -> MHSA -> Add&LayerNorm -> FFN -> Add&LayerNorm.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from positional_encoding import PositionalEncoding


def _check(name: str, tensor: Tensor, expected: tuple[int, ...], verbose: bool) -> None:
    assert tuple(tensor.shape) == expected, f"{name}: got {tuple(tensor.shape)}, expected {expected}"
    if verbose:
        print(f"  {name:<22s} {tuple(tensor.shape)}")


class MultiHeadSelfAttention(nn.Module):
    """Linear Q/K/V -> split heads -> scaled dot-product -> softmax -> concat -> W^O."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        # Mot Linear(d_model, d_model) = n_heads phep chieu (d_model, d_k) xep canh nhau.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None, return_attn: bool = False, verbose: bool = False):
        b, t, d = x.shape
        h, dk = self.n_heads, self.d_k
        _check("X", x, (b, t, d), verbose)
        q, k, v = self._split_heads(self.w_q(x)), self._split_heads(self.w_k(x)), self._split_heads(self.w_v(x))
        for name, m in (("Q", q), ("K", k), ("V", v)):
            _check(name, m, (b, h, t, dk), verbose)

        scores = q @ k.transpose(-2, -1) / math.sqrt(dk)  # (B, h, T, T)
        _check("scores QK^T/sqrt(dk)", scores, (b, h, t, t), verbose)
        if pad_mask is not None:
            assert pad_mask.shape == (b, t), f"pad_mask shape {tuple(pad_mask.shape)}"
            scores = scores.masked_fill(pad_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        _check("attn weights", attn, (b, h, t, t), verbose)
        heads = self.attn_dropout(attn) @ v  # (B, h, T, d_k)
        _check("head outputs", heads, (b, h, t, dk), verbose)
        concat = heads.transpose(1, 2).contiguous().view(b, t, d)  # Concat(head_1..head_h)
        _check("concat heads", concat, (b, t, d), verbose)
        out = self.w_o(concat)
        _check("after W^O", out, (b, t, d), verbose)
        return (out, attn) if return_attn else out


class TransformerBlock(nn.Module):
    """Post-norm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, activation: str = "relu") -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.d_model, self.d_ff = d_model, d_ff

    def forward(self, x: Tensor, pad_mask: Tensor | None = None, return_attn: bool = False, verbose: bool = False):
        b, t, d = x.shape
        attn_out, attn = self.attn(x, pad_mask, return_attn=True, verbose=verbose)
        x = self.norm1(x + self.drop(attn_out))  # Add & Norm
        _check("Add&Norm 1", x, (b, t, d), verbose)
        hidden = self.ffn[1](self.ffn[0](x))
        _check("FFN hidden", hidden, (b, t, self.d_ff), verbose)
        ffn_out = self.ffn[3](self.ffn[2](hidden))
        _check("FFN out", ffn_out, (b, t, d), verbose)
        out = self.norm2(x + self.drop(ffn_out))  # Add & Norm
        _check("output", out, (b, t, d), verbose)
        return (out, attn) if return_attn else out


class TransformerClassifier(nn.Module):
    """Embedding -> (PE) -> N x TransformerBlock -> pooling -> Linear.

    pooling="mean": trung binh co mask padding.
    pooling="cls" : them 1 vector [CLS] hoc duoc vao dau chuoi, lay hidden state cua no.
    use_pe=False  : bo positional encoding (ablation, Exp 1).
    """

    def __init__(
        self, vocab_size: int, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, n_layers: int = 2,
        dropout: float = 0.1, pooling: str = "mean", use_pe: bool = True, max_len: int = 512, pad_index: int = 0,
    ) -> None:
        super().__init__()
        assert pooling in {"mean", "cls"}
        self.pooling, self.use_pe, self.pad_index = pooling, use_pe, pad_index
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_index)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if pooling == "cls" else None
        self.pos = PositionalEncoding(d_model, max_len + 1) if use_pe else None
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def encode(self, input_ids: Tensor, return_attn: bool = False):
        pad_mask = input_ids == self.pad_index
        x = self.embedding(input_ids)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            pad_mask = torch.cat([torch.zeros_like(pad_mask[:, :1]), pad_mask], dim=1)
        if self.pos is not None:
            x = self.pos(x)
        x = self.drop(x)
        attns = []
        for block in self.blocks:
            x, a = block(x, pad_mask, return_attn=True)
            attns.append(a)
        return (x, pad_mask, attns) if return_attn else (x, pad_mask)

    def forward(self, input_ids: Tensor, lengths: Tensor | None = None, return_attn: bool = False):
        x, pad_mask, attns = self.encode(input_ids, return_attn=True)
        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        logits = self.classifier(self.drop(pooled)).squeeze(-1)
        return (logits, attns) if return_attn else logits


# ----------------------------------------------------------------------------- tests
def check_shapes(verbose: bool = True) -> None:
    """Muc 18: B=8, T=100, d_model=128, n_heads=4 -> d_k=32."""
    torch.manual_seed(0)
    b, t, d, h = 8, 100, 128, 4
    block = TransformerBlock(d, h, 4 * d)
    x = torch.randn(b, t, d)
    if verbose:
        print(f"B={b} T={t} d_model={d} n_heads={h} d_k={d // h}")
    out, attn = block(x, return_attn=True, verbose=verbose)
    assert out.shape == (b, t, d) and attn.shape == (b, h, t, t)
    assert torch.allclose(attn.sum(-1), torch.ones(b, h, t), atol=1e-5)


def check_padding_mask() -> None:
    """Noi dung tai vi tri PAD khong duoc anh huong output tai vi tri hop le; attn toi PAD = 0."""
    torch.manual_seed(0)
    block = TransformerBlock(32, 4, 64).eval()
    x = torch.randn(2, 10, 32)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 6:] = True
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 32) * 5
    o1, a1 = block(x, mask, return_attn=True)
    o2, _ = block(x2, mask, return_attn=True)
    assert torch.allclose(o1[:, :6], o2[:, :6], atol=1e-5)
    assert a1[..., 6:].abs().max() == 0.0


def run_self_tests() -> None:
    check_shapes(verbose=True)
    check_padding_mask()
    print("transformer_block tests: OK")


if __name__ == "__main__":
    run_self_tests()
