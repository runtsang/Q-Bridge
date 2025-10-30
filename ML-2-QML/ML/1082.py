"""Hybrid graph neural network utilities with batch‑wise training.

This module extends the original seed by adding:

* A :class:`GraphQNNModel` that stores the architecture and a list of
  :class:`torch.nn.Linear` layers.
* A :func:`train_model` helper that runs a mini‑batch training loop using
  mean‑squared error loss and Adam optimiser.
* A :func:`batch_feedforward` that accepts a list of input tensors and
  returns a list of activation lists, matching the signature of the
  original :func:`feedforward` but with vectorised support.
* A :func:`fidelity_adjacency` that now accepts either a list of
  :class:`torch.Tensor` or :class:`qutip.Qobj` instances and dispatches
  to the appropriate fidelity routine.  The function also exposes a
  ``secondary_weight`` keyword to weight edges that fall between two
  thresholds.

The public API mirrors the seed while providing additional experimental
capabilities for quick prototyping.

"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor
QObj = "qutip.Qobj"  # forward reference

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (x, y) for a classical feed‑forward network."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight matrix for the last layer and generate
    a training dataset for it.  Returns the architecture, a list of
    weight matrices for every layer, the training data, and the target
    weight matrix for the final layer."""
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
) -> List[List[Tensor]]:
    """Run a forward pass for each sample and return the activations
    at every layer.  The interface matches the original seed."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def batch_feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    inputs: Iterable[Tensor],
) -> List[List[Tensor]]:
    """Vectorised feed‑forward that accepts a batch of input tensors.
    Returns a list of activation lists, one per sample."""
    return feedforward(qnn_arch, weights, ((x, None) for x in inputs))


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Union[Tensor, "qutip.Qobj"]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.
    Supports both classical tensors and qutip states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        if isinstance(state_i, torch.Tensor):
            fid = state_fidelity(state_i, state_j)  # type: ignore[arg-type]
        else:
            fid = abs((state_i.dag() * state_j)[0, 0]) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Model class
# --------------------------------------------------------------------------- #
class GraphQNNModel(nn.Module):
    """A simple feed‑forward graph neural network built from linear layers."""

    def __init__(self, architecture: Sequence[int]):
        super().__init__()
        self.architecture = list(architecture)
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def feedforward(self, inputs: Iterable[Tensor]) -> List[List[Tensor]]:
        """Return activations for each sample in the batch."""
        activations_list: List[List[Tensor]] = []
        for inp in inputs:
            activations = [inp]
            current = inp
            for layer in self.layers:
                current = torch.tanh(layer(current))
                activations.append(current)
            activations_list.append(activations)
        return activations_list


def train_model(
    model: GraphQNNModel,
    training_data: Iterable[Tuple[Tensor, Tensor]],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Train the model using MSE loss and Adam optimiser.
    Returns a list of average training loss per epoch."""
    device = device or torch.device("cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: List[float] = []

    data = list(training_data)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            inputs, targets = zip(*batch)
            inputs_t = torch.stack(inputs).to(device)
            targets_t = torch.stack(targets).to(device)

            optimizer.zero_grad()
            outputs = model(inputs_t)
            loss = criterion(outputs, targets_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
    return losses


__all__ = [
    "GraphQNNModel",
    "train_model",
    "feedforward",
    "batch_feedforward",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
