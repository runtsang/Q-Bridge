"""Enhanced classical LSTM with dropout, residuals, and hybrid‑quantum gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

__all__ = ["QLSTMGen"]


class QLSTMClassic(nn.Module):
    """Classic LSTM cell with optional dropout and residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.dropout(self.forget_linear(combined)))
            i = torch.sigmoid(self.dropout(self.input_linear(combined)))
            g = torch.tanh(self.dropout(self.update_linear(combined)))
            o = torch.sigmoid(self.dropout(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.use_residual:
                hx = hx + self.residual_proj(x)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class QLSTMGen(nn.Module):
    """Wrapper that exposes a unified API for classical LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = QLSTMClassic(input_dim, hidden_dim, dropout, use_residual)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(inputs, states)
