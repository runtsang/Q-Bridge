"""Hybrid EstimatorQNN with self‑attention and optional depth scaling.

This module defines EstimatorQNN__gen384, a PyTorch neural network that
combines a lightweight self‑attention block, a classical feed‑forward
backbone and an optional quantum feature extractor based on the
pennylane qnode defined in the sibling qml module.

Classes
-------
EstimatorQNN__gen384
    Main hybrid estimator.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type

class ClassicalAttention(nn.Module):
    """A lightweight self‑attention block that matches the quantum interface."""
    def __init__(self, embed_dim: int = 4, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.out_proj = nn.Linear(embed_dim * heads, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        B, N, E = x.shape
        q = self.q_proj(x).view(B, N, self.heads, E).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, E).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, E).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.heads * E)
        out = self.out_proj(out)
        return out


class EstimatorQNN__gen384(nn.Module):
    """
    Hybrid estimator that optionally injects a quantum block
    (defined in the qml module) after a self‑attention and a
    classical residual backbone.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input features.
    hidden_dim : int, default 384
        Width of the hidden layers and the size of the quantum register.
    depth : int, default 3
        Number of classical layers (and quantum ansatz depth if used).
    use_quantum : bool, default False
        If True, a quantum block is appended after the classical backbone.
    use_attention : bool, default True
        If True, a self‑attention block is applied to the projected features.
    quantum_class : Optional[type]
        The quantum class to instantiate when ``use_quantum=True``.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        depth: int = 3,
        use_quantum: bool = False,
        use_attention: bool = True,
        quantum_class: Optional[Type] = None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = ClassicalAttention(embed_dim=hidden_dim, heads=2, dropout=0.1)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.use_quantum = use_quantum
        if self.use_quantum:
            if quantum_class is None:
                raise ValueError("``quantum_class`` must be provided when use_quantum=True")
            self.quantum_block = quantum_class(num_qubits=hidden_dim, depth=depth)
        else:
            self.quantum_block = None
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        if self.use_attention:
            # Self‑attention expects a [B, seq, E] tensor; we treat the
            # features as a single “token” by expanding dimension.
            h = self.attention(h.unsqueeze(1)).squeeze(1)
        for layer in self.layers:
            h = layer(h)
        if self.quantum_block is not None:
            # quantum_block expects a 1‑D vector per example
            h = torch.stack([self.quantum_block(sample) for sample in h], dim=0)
        return self.output(h)

__all__ = ["EstimatorQNN__gen384", "ClassicalAttention"]
