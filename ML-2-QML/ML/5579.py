"""
CombinedAttentionClassifierGraphCNN
====================================

This module defines a PyTorch model that unites three building blocks
seen in the reference seeds:
* Classical self‑attention (mirrors the quantum SelfAttention circuit)
* Graph‑based feed‑forward (inspired by GraphQNN)
* QCNN‑style fully connected stack (inspired by QCNNModel)

The resulting :class:`CombinedModel` exposes a single ``forward`` method
and optional helpers for synthetic data generation and fidelity‑based
graph construction.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# 1. Classical self‑attention
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product self‑attention implemented in PyTorch."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


# --------------------------------------------------------------------------- #
# 2. Graph‑based feed‑forward
# --------------------------------------------------------------------------- #
class GraphFeedforward(nn.Module):
    """Feed‑forward network with an arbitrary layer width sequence."""

    def __init__(self, arch: List[int]):
        super().__init__()
        layers = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(arch[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))


# --------------------------------------------------------------------------- #
# 3. QCNN‑style fully‑connected stack
# --------------------------------------------------------------------------- #
class QCNNLike(nn.Module):
    """Emulates the QCNNModel from the reference seeds."""

    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# 4. The integrated model
# --------------------------------------------------------------------------- #
class CombinedModel(nn.Module):
    """Combines the three blocks into a single end‑to‑end model."""

    def __init__(self, embed_dim: int, gnn_arch: List[int]):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.gnn = GraphFeedforward(gnn_arch)
        self.cnn = QCNNLike()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention → GNN → QCNN."""
        attn_out = self.attention(x)
        gnn_out = self.gnn(attn_out)
        return self.cnn(gnn_out)


# --------------------------------------------------------------------------- #
# 5. Utility functions
# --------------------------------------------------------------------------- #
def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic data that targets a linear transformation."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset


def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = float((state_i / state_i.norm()) @ (state_j / state_j.norm()))
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


__all__ = [
    "ClassicalSelfAttention",
    "GraphFeedforward",
    "QCNNLike",
    "CombinedModel",
    "random_training_data",
    "fidelity_adjacency",
]
