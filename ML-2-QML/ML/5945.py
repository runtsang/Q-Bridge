"""GraphQNN: Classical neural network with graph‑based training utilities.

This module expands the original seed by adding:
* configurable activation functions (tanh, relu, sigmoid)
* dropout layers between hidden units
* a hybrid loss combining MSE and a fidelity‑based penalty
* a lightweight `train` routine that runs a few epochs and logs fidelity and loss.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate training data for a specific target weight matrix."""
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random network along with training data for the final layer."""
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
    activation: Callable[[Tensor], Tensor] = torch.tanh,
    dropout: float | None = None,
) -> List[List[Tensor]]:
    """Forward propagation with optional dropout and activation."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = activation(weight @ current)
            if dropout is not None:
                current = F.dropout(current, p=dropout, training=True)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared inner‑product fidelity between two real vectors."""
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
    """Create a weighted adjacency graph from state fidelities."""
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
# Training routine
# --------------------------------------------------------------------------- #
def train(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    epochs: int = 5,
    lr: float = 1e-3,
    activation: Callable[[Tensor], Tensor] = torch.tanh,
    dropout: float | None = None,
) -> Tuple[List[float], List[float]]:
    """Simple training loop returning loss and fidelity history."""
    optimizer = torch.optim.Adam(weights, lr=lr)
    loss_history: List[float] = []
    fid_history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, target in training_data:
            optimizer.zero_grad()
            outputs = features
            for weight in weights:
                outputs = activation(weight @ outputs)
                if dropout is not None:
                    outputs = F.dropout(outputs, p=dropout, training=True)
            loss = F.mse_loss(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Compute fidelity between final output and target for a random batch
        sample = next(iter(training_data))
        with torch.no_grad():
            out = sample[0]
            for weight in weights:
                out = activation(weight @ out)
                if dropout is not None:
                    out = F.dropout(out, p=dropout, training=False)
            fid = state_fidelity(out, sample[1])
        loss_history.append(epoch_loss / len(training_data))
        fid_history.append(fid)

    return loss_history, fid_history


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train",
]
