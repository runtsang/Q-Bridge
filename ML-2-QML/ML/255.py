"""
GraphQNN: Classical neural network with graph‑based fidelity metrics.

This module extends the original seed by adding:
* A small feed‑forward network that can be trained on a classical optimiser.
* A loss function that compares the target weight matrix with the output of the
  network (MSE).
* A helper that builds a graph from the (fidelity‑based) similarity of
  the training samples.
* Utility functions for generating synthetic data and random weights.

The interface is deliberately kept compatible with the seed – the public
functions are the same names – but the internal implementation now uses
PyTorch and supports stochastic‑gradient descent.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss

Tensor = torch.Tensor

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_network",
    "graph_from_samples",
]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with a fixed seed for reproducibility."""
    torch.manual_seed(42)
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> list[tuple[Tensor, Tensor]]:
    """Generate a dataset where the input is a random vector and the target is
    the matrix‑vector product with ``weight``.
    """
    dataset: list[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> tuple[list[int], list[Tensor], list[tuple[Tensor, Tensor]], Tensor]:
    """Return an architecture, a list of random weight tensors, a training
    dataset and the target weight matrix (the last layer).
    """
    weights: list[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> list[list[Tensor]]:
    """Run a forward pass through the network and return all intermediate
    activations.  The activations are stored layer‑wise for each sample
    and are useful for graph construction.
    """
    stored: list[list[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two normalised vectors."""
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
    """Create a weighted graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
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


def train_network(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    training_data: Iterable[tuple[Tensor, Tensor]],
    target_weight: Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
) -> list[Tensor]:
    """Train a simple feed‑forward network with SGD to minimise the
    mean‑squared error between the network output and ``target_weight``.
    """
    # Build a PyTorch model that mirrors the weight list
    layers = nn.ModuleList()
    for w in weights:
        linear = nn.Linear(w.size(1), w.size(0), bias=False)
        linear.weight.data = w.clone()
        layers.append(linear)

    model = nn.Sequential(*layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, y in training_data:
            optimizer.zero_grad()
            out = x
            for layer in model:
                out = torch.tanh(layer(out))
            loss = mse_loss(out, target_weight @ x)
            loss.backward()
            optimizer.step()

    # Return the trained weight tensors
    trained_weights = [layer.weight.data.clone() for layer in model]
    return trained_weights


def graph_from_samples(
    samples: Iterable[tuple[Tensor, Tensor]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Convenience wrapper that builds a graph from the activations of a
    feed‑forward network.  It extracts the last hidden layer from each sample
    and constructs the graph using ``fidelity_adjacency``.
    """
    activations = feedforward(
        qnn_arch=[s[0].shape[0] for s in samples],  # dummy arch
        weights=[_random_linear(s[0].shape[0], s[0].shape[0]) for _ in samples],
        samples=samples,
    )
    last_layer_states = [acts[-1] for acts in activations]
    return fidelity_adjacency(
        last_layer_states,
        threshold,
        secondary=secondary,
        secondary_weight=secondary_weight,
    )
