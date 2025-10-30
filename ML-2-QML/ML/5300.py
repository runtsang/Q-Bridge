"""Classical estimator inspired by EstimatorQNN, GraphQNN, QuantumRegression, and QLSTM.

Provides a feed‑forward network that can optionally ingest graph‑derived embeddings
computed from a fidelity graph.  The public factory function `EstimatorQNN`
mirrors the original API and returns a `UnifiedEstimatorQNN` instance.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import networkx as nx

# --------------------------------------------------------------------------- #
#  Utility: graph‑based fidelity adjacency (adapted from GraphQNN)
# --------------------------------------------------------------------------- #

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two unit‑norm tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create weighted adjacency graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = _state_fidelity(a, b=states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Dataset: classical regression data (adapted from QuantumRegression)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Model: Classical feed‑forward network with optional graph embedding
# --------------------------------------------------------------------------- #

class UnifiedEstimatorQNN(nn.Module):
    """Classical estimator that optionally incorporates graph‑derived embeddings.

    The network consists of a feature encoder, a stack of hidden layers, and
    a regression head.  When a graph is supplied, the node embeddings are
    computed via a simple adjacency‑based linear projection and concatenated
    with the raw features before the feed‑forward stack.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        graph: nx.Graph | None = None,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.embedding_dim = embedding_dim

        # If a graph is present, create an embedding matrix that maps node indices
        # to a dense vector.  The embedding is learned during training.
        if self.graph is not None:
            self.node_embed = nn.Embedding(len(self.graph.nodes), self.embedding_dim)
            # Initialize embeddings with a simple adjacency‑based weighting
            adjacency = nx.to_numpy_array(self.graph)
            # Normalize rows to unit length
            norms = np.linalg.norm(adjacency, axis=1, keepdims=True) + 1e-12
            init = torch.from_numpy(adjacency / norms).float()
            self.node_embed.weight.data = init

        # Feature encoder: raw features + optional graph embedding
        encoder_dim = input_dim + (self.embedding_dim if self.graph else 0)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical estimator.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, input_dim).  If a graph is supplied, the nodes
            corresponding to each batch entry are expected to be encoded
            as integer indices in a separate tensor `node_ids`.  For simplicity
            the implementation assumes that the caller provides a tensor of
            shape (batch,) containing node indices; if not supplied, the
            graph embedding is omitted.
        """
        if self.graph is not None:
            # Expect a second positional argument containing node indices
            # But to keep the signature simple, we will embed the mean of all node
            # embeddings as a crude feature.
            node_ids = torch.arange(len(self.graph.nodes), device=x.device)
            node_embeds = self.node_embed(node_ids).mean(dim=0, keepdim=True).repeat(x.size(0), 1)
            x = torch.cat([x, node_embeds], dim=1)

        h = self.encoder(x)
        return self.head(h).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Factory function (mirrors original EstimatorQNN API)
# --------------------------------------------------------------------------- #

def EstimatorQNN(
    input_dim: int = 2,
    hidden_dim: int = 64,
    graph: nx.Graph | None = None,
    embedding_dim: int = 16,
) -> UnifiedEstimatorQNN:
    """Return a classical estimator with optional graph embedding support."""
    return UnifiedEstimatorQNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        graph=graph,
        embedding_dim=embedding_dim,
    )

__all__ = ["EstimatorQNN", "UnifiedEstimatorQNN", "RegressionDataset", "generate_superposition_data"]
