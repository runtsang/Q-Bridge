"""
HybridFusionNetQuantum: Quantum‑enhanced counterpart using Pennylane for expectation‑based layers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


# --------------------------------------------------------------------------- #
#  Quantum utilities – simple variational circuits
# --------------------------------------------------------------------------- #

class QuantumExpectation(nn.Module):
    """
    Maps a scalar to an expectation value of a single‑qubit circuit.
    Parameterised by a rotation angle; differentiable via Pennylane autograd.
    """
    def __init__(self, shift: float = 0.0, shots: int = 1000):
        super().__init__()
        self.shift = shift
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=1, shots=shots)
        self.params = nn.Parameter(torch.tensor([0.0]))

    def circuit(self, theta, w):
        qml.Hadamard(w)
        qml.RX(theta, w)
        return qml.expval(qml.PauliZ(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a scalar per sample
        out = qml.QNode(self.circuit, self.dev)(self.params + self.shift, 0)
        return torch.tensor(out, dtype=x.dtype, device=x.device)


class QuantumHybridHead(nn.Module):
    """
    Dense layer followed by a QuantumExpectation head.
    """
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.exp = QuantumExpectation(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.exp(self.fc(x))


# --------------------------------------------------------------------------- #
#  Transformer‑style attention with quantum heads
# --------------------------------------------------------------------------- #

class QuantumAttentionLayer(nn.Module):
    """
    Implements a miniature attention head using a variational circuit
    to produce attention scores for each token.
    """
    def __init__(self, embed_dim: int, num_heads: int, shots: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dev = qml.device("default.qubit", wires=num_heads, shots=shots)
        self.params = nn.Parameter(torch.randn(num_heads, 1))

    def circuit(self, theta, w):
        qml.Hadamard(w)
        qml.RX(theta, w)
        return qml.expval(qml.PauliZ(w))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x shape: (B, T, E)
        B, T, E = x.shape
        # project to heads
        proj = x @ torch.randn(E, self.num_heads, device=x.device)
        scores = []
        for h in range(self.num_heads):
            out = qml.QNode(self.circuit, self.dev)(self.params[h] + proj[:, :, h], h)
            scores.append(out)
        scores = torch.stack(scores, dim=2)  # (B, T, H)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=1)
        return attn


class QuantumTransformerBlock(nn.Module):
    """
    A transformer block that uses the QuantumAttentionLayer and a
    quantum‑based feed‑forward expansion.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 shots: int = 1000):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttentionLayer(embed_dim, num_heads, shots)
        # feed‑forward via a tiny variational circuit
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attn(x)
        attn_out = torch.sum(attn_scores.unsqueeze(-1) * x, dim=1, keepdim=True)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# --------------------------------------------------------------------------- #
#  Final quantum‑enhanced classifier
# --------------------------------------------------------------------------- #

class HybridFusionNetQuantum(nn.Module):
    """
    Quantum‑enhanced fusion model that mirrors HybridFusionNet but
    replaces the head with a QuantumHybridHead and the transformer
    with QuantumTransformerBlock(s).
    """
    def __init__(self,
                 embed_dim: int = 120,
                 num_heads: int = 8,
                 ffn_dim: int = 256,
                 num_blocks: int = 2,
                 shots: int = 1000):
        super().__init__()
        self.backbone = ConvBackbone()  # same as classical version
        self.transformer = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, shots)
              for _ in range(num_blocks)]
        )
        self.head = QuantumHybridHead(in_features=embed_dim, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        flattened = torch.flatten(x, 1)
        # Project to embed_dim via a linear layer (same as classical)
        proj = nn.Linear(flattened.shape[1], embed_dim).to(x.device)(flattened)
        x = self.transformer(proj)
        return self.head(x).squeeze(-1)


__all__ = [
    "QuantumExpectation",
    "QuantumHybridHead",
    "QuantumAttentionLayer",
    "QuantumTransformerBlock",
    "HybridFusionNetQuantum",
]
