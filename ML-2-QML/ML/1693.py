"""GraphQNN with a trainable two‑stage feedforward network.

This module builds on the original seed by adding:
* A **trainable linear layer** after the first activation to allow the model to learn a mapping from
  hidden activations to the target.
* A **`train_model` method** that optimises the weight matrix using Adam and records loss history.
* A **`predict` helper** that runs the full forward pass and returns the final tensor.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    """Return a PyTorch linear layer with random weights and bias."""
    return nn.Linear(in_features, out_features, bias=bias)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset of random inputs and targets using the provided weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Instantiate a random feed‑forward network and a training set."""
    weights: List[nn.Linear] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1].weight
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[nn.Linear],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass and capture intermediate activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight(current))
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Compute the squared overlap between two pure states."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((torch.dot(a_norm, b_norm)).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNModel(nn.Module):
    """A two‑stage classical feed‑forward network with a trainable final layer."""

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList(
            [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out

    def train_model(
        self,
        X: Tensor,
        Y: Tensor,
        lr: float = 1e-3,
        epochs: int = 200,
        verbose: bool = False,
    ) -> List[float]:
        """Optimise the network to minimise MSE on the provided data."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loss_history: List[float] = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {loss.item():.6f}")

        return loss_history

    def predict(self, X: Tensor) -> Tensor:
        """Return the network’s output for the provided inputs."""
        with torch.no_grad():
            return self.forward(X)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNModel",
]
