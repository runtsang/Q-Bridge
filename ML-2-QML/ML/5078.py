"""HybridEstimator: classical core + quantum‑augmented layer + self‑attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SelfAttention(nn.Module):
    """Classical self‑attention used in the hybrid estimator."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, v)


class QuantumApproxLayer(nn.Module):
    """Surrogate for a one‑qubit quantum circuit."""
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.sin(self.param * x)
        out = out * self.scale + self.shift
        return out


class HybridEstimator(nn.Module):
    """Hybrid classical‑quantum estimator."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_attention: bool = False,
        use_lstm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention
        self.use_lstm = use_lstm

        self.core = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.quantum = QuantumApproxLayer()

        self.attention = SelfAttention(embed_dim=hidden_dim) if use_attention else None
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True) if use_lstm else None

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            core_out = self.core(x)
        else:
            batch, seq_len, _ = x.shape
            core_out = self.core(x.view(batch * seq_len, -1))
            core_out = core_out.view(batch, seq_len, -1)

        q_out = self.quantum(core_out)

        out = core_out + q_out

        if self.attention is not None:
            out = self.attention(out)

        if self.lstm is not None:
            out, _ = self.lstm(out)

        out = self.output(out)

        if x.dim() == 2:
            out = out.squeeze(-1)
        return out


__all__ = ["HybridEstimator"]
