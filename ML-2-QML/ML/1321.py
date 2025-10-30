"""GraphQNN: classical neural network with a lightweight training pipeline.

The new module keeps the original API (`feedforward`, `fidelity_adjacency`,
`random_network`, `random_training_data`, `state_fidelity`) but adds a
`train` helper that runs a mean‑square loss optimisation with Adam.
The implementation uses PyTorch, so the network is differentiable and can
be trained on CPU or GPU.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Helper functions – unchanged from the seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate training data for a single linear layer.

    Each sample is a pair ``(features, target)`` where
    ``target = weight @ features``.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and a training set for the last layer.

    Returns:
        arch: list[int]
        weights: list[Tensor] – one weight matrix per layer
        training_data: list[tuple[Tensor, Tensor]]
        target_weight: Tensor – the last layer’s weight matrix
    """
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return activations for each sample through the network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two state vectors."""
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
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= threshold receive weight 1.0.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
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
# GraphQNN class – adds a simple training loop
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """A lightweight feed‑forward network that mirrors the original API.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 2]``.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        layers = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass with tanh activations."""
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def train_network(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> List[float]:
        """Train the network with MSE loss and Adam optimiser.

        Returns
        -------
        losses : list[float]
            Training loss per epoch.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for features, target in training_data:
                optimizer.zero_grad()
                output = self(features)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(training_data)
            losses.append(epoch_loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.6f}")
        return losses


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
