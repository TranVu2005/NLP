"""A minimal, manually implemented one-step tanh RNN cell."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SimpleRNNCell(nn.Module):
    """Compute one tanh recurrent step without ``nn.RNNCell``.

    ``weight_ih`` follows PyTorch's ``(hidden_size, input_size)`` layout;
    ``weight_hh`` has shape ``(hidden_size, hidden_size)``.
    """

    def __init__(self, input_size: int, hidden_size: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size, dtype=dtype))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=dtype))
        self.bias_h = nn.Parameter(torch.empty(hidden_size, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        nn.init.uniform_(self.bias_h, -bound, bound)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        if x_t.shape[-1] != self.input_size:
            raise ValueError(f"Expected input's last dimension to be {self.input_size}, got {x_t.shape[-1]}")
        if h_prev.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden state's last dimension to be {self.hidden_size}, got {h_prev.shape[-1]}")
        if x_t.shape[:-1] != h_prev.shape[:-1]:
            raise ValueError("Input and previous hidden state must have matching batch dimensions")
        return torch.tanh(F.linear(x_t, self.weight_ih) + F.linear(h_prev, self.weight_hh) + self.bias_h)
