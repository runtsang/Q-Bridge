"""Unified classical regression framework with graph‑based and quantum‑aware extensions.

The module defines:
* :class:`UnifiedQuantumRegression` – a hybrid model that first encodes inputs via a classical linear layer,
  then refines them with a graph‑based layer, and finally produces a regression output with an MLP head.
* :class:`UnifiedDataset` – synthetic dataset generator based on superposition data.
* :class:`GraphQLayer` – a lightweight graph neural network that operates on embeddings.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

# Data generation utilities
def generate_superposition_data(num_features: int, samples: int, *,
                                noise: float = 0.1, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + noise * np.cos(2 * angles)
    return x, y.astype(np.float32)

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int, *, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, seed=seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# Classical MLP head
class MLPHead(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: Sequence[int] = (64, 32), out_features: int = 1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# Graph‑based quantum layer (classical version)
class GraphQLayer(nn.Module):
    """
    Operates on a batch of node embeddings.  Edges are constructed from
    the squared dot‑product (fidelity) between embeddings.  The layer
    aggregates neighbor embeddings weighted by edge weights.
    """
    def __init__(self, node_features: int, hidden_size: int = 32, graph_threshold: float = 0.85):
        super().__init__()
        self.node_features = node_features
        self.graph_threshold = graph_threshold
        self.linear = nn.Linear(node_features, hidden_size)

    def _build_graph(self, embeddings: torch.Tensor) -> nx.Graph:
        n = embeddings.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                fid = torch.dot(embeddings[i], embeddings[j]).item() ** 2
                weight = 1.0 if fid >= self.graph_threshold else 0.5
                if fid >= self.graph_threshold:
                    g.add_edge(i, j, weight=weight)
                elif self.graph_threshold < fid < self.graph_threshold + 0.1:
                    g.add_edge(i, j, weight=weight)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self._build_graph(x)
        agg = torch.zeros_like(x)
        for node in g.nodes:
            neighbors = list(g.neighbors(node))
            if neighbors:
                weights = torch.tensor([g[node][nbr]["weight"] for nbr in neighbors], device=x.device)
                neighbor_feats = x[neighbors]
                agg[node] = (weights @ neighbor_feats).mean()
        return self.linear(agg)

# Unified model
class UnifiedQuantumRegression(nn.Module):
    """
    Hybrid model that first encodes inputs via a classical linear layer,
    then refines them with a graph‑based layer, and finally produces a
    regression output with an MLP head.
    """
    def __init__(self,
                 num_features: int,
                 graph_hidden: int = 32,
                 mlp_hidden_sizes: Sequence[int] = (64, 32)):
        super().__init__()
        self.encoder = nn.Linear(num_features, num_features)  # placeholder for quantum encoder
        self.graph = GraphQLayer(num_features, hidden_size=graph_hidden)
        self.head = MLPHead(num_features, hidden_sizes=mlp_hidden_sizes)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state_batch)
        graph_feat = self.graph(encoded)
        out = self.head(graph_feat)
        return out

__all__ = ["UnifiedQuantumRegression", "UnifiedDataset", "GraphQLayer", "MLPHead", "generate_superposition_data"]
