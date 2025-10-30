"""Classical graph neural network utilities and regression dataset.

This module extends the original GraphQNN utilities by adding a
graph‑aware training pipeline and a regression dataset inspired
by the quantum regression example.  The design keeps deterministic
feed‑forward and fidelity‑graph construction while providing a
classical neural‑network backbone that can later be used in a hybrid
training loop.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
State = torch.Tensor  # alias for clarity

# --------------------------------------------------------------------------- #
# 1. Utility: Random graph / network generation
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_network(arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Generate a random linear network and a dataset for the last layer.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 1]``.
    samples : int
        Number of training samples.

    Returns
    -------
    arch, weights, dataset, target_weight
    """
    weights: List[Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1), dtype=torch.float32)
        labels = target_weight @ features
        dataset.append((features, labels))
    return arch, weights, dataset, target_weight


# --------------------------------------------------------------------------- #
# 2. Forward propagation helpers
# --------------------------------------------------------------------------- #
def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of activation tensors for each sample."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_outputs = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer_outputs.append(current)
        activations.append(layer_outputs)
    return activations


# --------------------------------------------------------------------------- #
# 3. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Compute the squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted adjacency graph based on state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 4. Classical graph‑aware neural network
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """A lightweight graph‑aware GNN that can be used as a classical backbone
    in a hybrid training loop.

    Parameters
    ----------
    arch : Sequence[int]
        Sizes of each layer.  The number of edges in the graph is inferred
        from the adjacency of the output states.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out.squeeze(-1)

    def fit(
        self,
        data_loader: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        """Simple SGD training loop."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for features, target in data_loader:
                features, target = features.to(device), target.to(device)
                pred = self(features)
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# --------------------------------------------------------------------------- #
# 5. Regression dataset (classical)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data that mimics the quantum superposition used in the
    quantum regression example but purely classically.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a feature vector and a target scalar."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


__all__ = [
    "GraphQNNHybrid",
    "generate_superposition_data",
    "RegressionDataset",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
