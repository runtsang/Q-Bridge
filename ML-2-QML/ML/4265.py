"""AdvancedQCNN: hybrid classical‑quantum convolutional neural network with attention and kernel modules."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Classical convolution‑like block
class _ClassicalConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, act: nn.Module = nn.Tanh()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))

# Classical self‑attention helper
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x @ self.rotation_params
        key = x @ self.entangle_params
        scores = F.softmax(query @ key.t() / math.sqrt(self.embed_dim), dim=-1)
        return scores @ x

# Classical RBF kernel module
class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (N, D), y: (P, D)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, P, D)
        dist2 = (diff ** 2).sum(-1)  # (N, P)
        return torch.exp(-self.gamma * dist2)

class AdvancedQCNN(nn.Module):
    """
    Hybrid model combining:
      * a classical convolution‑like backbone,
      * a self‑attention block that can be mapped to a quantum circuit,
      * an RBF kernel module for similarity evaluation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Sequence[int] = (16, 12, 8, 4),
        embed_dim: int = 4,
        kernel_gamma: float = 1.0,
        prototype_count: int = 10,
    ):
        super().__init__()
        # Feature map
        self.feature_map = nn.Linear(input_dim, hidden_dims[0])
        self.activation = nn.Tanh()

        # Classical convolution layers
        self.conv_layers = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.conv_layers.append(_ClassicalConvBlock(in_dim, out_dim))
            in_dim = out_dim

        # Self‑attention
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)

        # Kernel
        self.kernel = Kernel(gamma=kernel_gamma)

        # Prototypes and output
        self.prototypes = nn.Parameter(torch.randn(prototype_count, in_dim))
        self.out = nn.Linear(prototype_count, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Feature map
          2. Convolution layers
          3. Self‑attention
          4. Kernel similarity with prototypes
          5. Output
        """
        # 1
        h = self.activation(self.feature_map(x))

        # 2
        for layer in self.conv_layers:
            h = layer(h)

        # 3
        h = self.attention(h)

        # 4
        k = self.kernel(h, self.prototypes)  # (N, P)

        # 5
        out = self.out(k)
        return torch.sigmoid(out)

__all__ = ["AdvancedQCNN"]
