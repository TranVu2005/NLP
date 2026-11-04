"""Sinusoidal positional encoding (LAB 07, muc 9-10), viet tay theo cong thuc de bai.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Chay `python positional_encoding.py` de chay unit test d=4, pos=0,1,2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> Tensor:
    """Return PE of shape (max_len, d_model); d_model must be even."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sin/cos pairs")
    pe = torch.zeros(max_len, d_model, dtype=torch.float64)
    for pos in range(max_len):
        for i in range(d_model // 2):
            angle = pos / (10000.0 ** (2 * i / d_model))
            pe[pos, 2 * i] = math.sin(angle)
            pe[pos, 2 * i + 1] = math.cos(angle)
    return pe.to(torch.float32)


class PositionalEncoding(nn.Module):
    """X' = X + PE[:T]; PE is a fixed (non-trainable) buffer."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        if t > self.pe.shape[0]:
            raise ValueError(f"sequence length {t} exceeds max_len {self.pe.shape[0]}")
        return x + self.pe[:t].unsqueeze(0)


# Gia tri tham chieu tinh tay (d=4): pos 0, 1, 2
EXPECTED_D4 = torch.tensor(
    [
        [0.0, 1.0, 0.0, 1.0],
        [0.8415, 0.5403, 0.0100, 0.99995],
        [0.9093, -0.4161, 0.0200, 0.9998],
    ]
)


def test_pe_d4() -> torch.Tensor:
    pe = sinusoidal_positional_encoding(3, 4)
    assert pe.shape == (3, 4)
    assert torch.allclose(pe, EXPECTED_D4, atol=5e-5), f"PE mismatch:\n{pe}\nvs\n{EXPECTED_D4}"
    return pe


def test_pe_properties() -> None:
    pe = sinusoidal_positional_encoding(64, 16)
    # moi vi tri co vector khac nhau
    assert torch.cdist(pe, pe).fill_diagonal_(1.0).min() > 1e-3
    # |PE| <= 1
    assert pe.abs().max() <= 1.0 + 1e-6
    # module cong dung vao dau vao
    x = torch.zeros(2, 5, 16)
    assert torch.allclose(PositionalEncoding(16, 64)(x), pe[:5].expand(2, -1, -1))


def run_self_tests() -> None:
    pe = test_pe_d4()
    test_pe_properties()
    print("PE d=4, pos=0,1,2:")
    print(pe.numpy().round(5))
    print("positional_encoding tests: OK")


if __name__ == "__main__":
    run_self_tests()
