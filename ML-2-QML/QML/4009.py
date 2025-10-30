"""Quantum‑enhanced transformer block using Qiskit.

This module implements a SelfAttentionTransformer that replaces the
classical attention sub‑module with a quantum circuit.  The quantum
circuit is executed on the Aer qasm simulator and returns a probability
that is used as the attention weight.  The rest of the transformer
(FFN, LayerNorm, etc.) remains classical, making the whole network
compatible with standard PyTorch training loops.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Aer, execute

__all__ = ["SelfAttentionTransformer"]


class QuantumSelfAttention(nn.Module):
    """Multi‑head attention where the similarity score is produced
    by a small Qiskit circuit.  The circuit contains a single qubit
    whose rotation angle encodes the dot product of two token
    embeddings.  The measurement probability of |1> is used as the
    attention weight.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits: int = 8, seed: int | None = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.backend = Aer.get_backend("qasm_simulator")
        self.seed = seed

    def _quantum_weight(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Return a probability based on the dot product of two vectors."""
        # Compute dot product and map to a rotation angle in [-π, π]
        dot = torch.sum(vec1 * vec2, dim=-1).item()
        angle = float(dot)  # simple linear mapping; can be scaled if desired
        angle = max(-math.pi, min(math.pi, angle))  # clamp
        qc = QuantumCircuit(1, 1)
        qc.rx(angle, 0)
        qc.measure(0, 0)
        job = execute(qc, self.backend, shots=1024, seed_simulator=self.seed)
        counts = job.result().get_counts()
        prob_one = counts.get("1", 0) / 1024
        return prob_one

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        # compute attention matrix
        attn = torch.zeros(B, self.num_heads, L, L, device=x.device, dtype=torch.float32)
        for i in range(L):
            for j in range(L):
                # For each pair of positions, compute quantum weight
                vec_i = x[:, :, i, :]  # (B, H, d_k)
                vec_j = x[:, :, j, :]  # (B, H, d_k)
                # average over batch and heads to obtain a scalar per pair
                weight = self._quantum_weight(vec_i.mean(dim=(0, 1)),
                                              vec_j.mean(dim=(0, 1)))
                attn[:, :, i, j] = weight
        # apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        # apply attention to values
        v = x  # use the same representation as key/value
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return out


class FeedForwardClassical(nn.Module):
    """Classical feed‑forward block used after the quantum attention."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SelfAttentionTransformer(nn.Module):
    """Transformer block with quantum‑based self‑attention."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_qubits: int = 8,
                 seed: int | None = None) -> None:
        super().__init__()
        self.attn = QuantumSelfAttention(embed_dim, num_heads, dropout,
                                         n_qubits=n_qubits, seed=seed)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))
