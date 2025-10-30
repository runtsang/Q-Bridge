"""Hybrid classical classifier integrating self‑attention and fully‑connected emulation."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn


class SelfAttention(nn.Module):
    """Classical self‑attention layer mimicking a quantum interface."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.value = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ self.query_weight
        key = inputs @ self.key_weight
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ self.value


class FCL(nn.Module):
    """Fully‑connected layer emulation with a single linear transform."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        thetas_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(thetas_tensor)).mean(dim=0)


class HybridClassifier(nn.Module):
    """Combined classical feed‑forward, self‑attention and FCL network."""
    def __init__(self,
                 num_features: int,
                 depth: int,
                 use_attention: bool = True,
                 attention_dim: int = 4,
                 use_fcl: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        if use_attention:
            layers.append(SelfAttention(embed_dim=attention_dim))
        if use_fcl:
            layers.append(FCL(in_features=num_features))
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)
        self.use_attention = use_attention
        self.use_fcl = use_fcl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, list[int], list[int], list[int]]:
        """Return the network and metadata analogous to the quantum interface."""
        net = HybridClassifier(num_features, depth)
        encoding = list(range(num_features))
        weight_sizes = [m.weight.numel() + m.bias.numel() for m in net.network if isinstance(m, nn.Linear)]
        observables = list(range(2))
        return net, encoding, weight_sizes, observables


__all__ = ["HybridClassifier"]
