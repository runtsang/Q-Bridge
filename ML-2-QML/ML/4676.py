"""Hybrid classical regression model with graph‑based feature weighting.

The module defines:
* `generate_superposition_data` – reproducible data generator used for both ML and QML.
* `RegressionDataset` – a torch Dataset yielding feature vectors and targets.
* `QModel` – a neural network that optionally uses a graph adjacency matrix derived from feature similarity.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic regression dataset.

    Features are uniformly sampled in [-1, 1].  The target is a non‑linear
    function of the sum of angles, mirroring the quantum seed.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset wrapping the synthetic data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Graph utilities – adapted from the GraphQNN seed – to build an adjacency
# matrix based on a fidelity‑like similarity measure.

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑norm vectors."""
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
    """Return a weighted graph whose nodes are the feature vectors.

    Edges are added when the fidelity exceeds ``threshold``; a secondary
    weaker connection can also be added.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid neural network that optionally incorporates the graph adjacency.

class QModel(nn.Module):
    """Graph‑aware feed‑forward regression network.

    Parameters
    ----------
    num_features : int
        Dimensionality of each input sample.
    hidden_dims : Sequence[int]
        Sizes of successive hidden layers.
    adjacency : nx.Graph | None
        If provided, the adjacency matrix is used to weight the
        hidden representation.  This is a lightweight way to inject
        graph structure without a full GNN implementation.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: Sequence[int] = (32, 16),
        adjacency: nx.Graph | None = None,
    ):
        super().__init__()
        self.adjacency = adjacency

        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(state_batch)
        if self.adjacency is not None:
            # Convert graph to adjacency matrix and perform a simple
            # weighted aggregation of the hidden representation.
            mat = nx.to_numpy_array(self.adjacency, dtype=np.float32)
            # Broadcast to batch dimension
            mat = torch.from_numpy(mat).to(state_batch.device)
            x = x @ mat.t()
        return x.squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "fidelity_adjacency"]
