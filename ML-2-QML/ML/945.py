"""GraphQNN: Classical MLP-based graph neural network with extended utilities.

This module extends the original seed by providing a trainable multi‑layer perceptron that
approximates the quantum layer outputs.  It retains the original feed‑forward, fidelity
and adjacency helpers while adding a convenient `fit`/`predict` interface.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate pairs (x, y) where y = Wx for a fixed linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random MLP and a training set for its last layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two state vectors."""
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
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN:
    """Hybrid classical graph neural network with a learnable MLP backbone.

    The class exposes a `fit` method that trains a multi‑layer perceptron to predict the
    output of a quantum‑style feed‑forward step.  The network is compatible with
    the original `feedforward`, `state_fidelity` and `fidelity_adjacency` helpers
    and can be used as a drop‑in replacement for the seed module.
    """

    def __init__(self, qnn_arch: Sequence[int], hidden_layers: Sequence[int] | None = None, device: str = "cpu"):
        self.qnn_arch = list(qnn_arch)
        self.device = torch.device(device)
        if hidden_layers is None:
            hidden_layers = self.qnn_arch[1:-1]
        self.hidden_layers = list(hidden_layers)
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_features = self.qnn_arch[0]
        for hidden in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.Tanh())
            in_features = hidden
        layers.append(nn.Linear(in_features, self.qnn_arch[-1]))
        return nn.Sequential(*layers)

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """Train the MLP using mean‑squared error."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: Tensor) -> Tensor:
        """Return the model output for the given inputs."""
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return the activations of each layer for every sample."""
        activations_per_sample: List[List[Tensor]] = []
        for features, _ in samples:
            layer_inputs = [features.to(self.device)]
            current = features.to(self.device)
            for layer in self.model:
                current = layer(current)
                layer_inputs.append(current)
            activations_per_sample.append([a.cpu() for a in layer_inputs])
        return activations_per_sample

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a random MLP and a training set for its last layer."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        """Generate random training data for a linear map."""
        return random_training_data(weight, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
