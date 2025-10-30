"""Graph-structured neural network with supervised readout and training.

This module extends the original seed by adding a trainable readout layer
and a simple MLP that ends with a softmax.  It retains the original
random_network and feedforward helpers but now supports a supervised
training loop.  The public API remains compatible with the seed – the
functions are still importable directly – while a GraphQNN class
provides convenient predict and train methods.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Randomly initialise a weight matrix with a standard normal."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a supervised dataset for a one‑layer MLP.

    The target is the linear transform of the input features.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        # Convert to a one‑hot vector for classification
        _, idx = torch.max(target, dim=0)
        target = F.one_hot(idx, num_classes=weight.shape[0]).float()
        dataset.append((features, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> tuple[list[int], list[torch.Tensor], List[Tuple[Tensor, Tensor]], torch.Tensor]:
    """Create a random MLP architecture and training data.

    The returned ``weights`` list contains *all* layers, including
    the final readout.
    """
    weights: List[Tensor] = []
    # Hidden layers
    for in_dim, out_dim in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_dim, out_dim))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass through the network.

    The output is a list of per‑layer activations for each sample.
    """
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
    """Return the squared overlap between two feature vectors."""
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

    Edges with fidelity greater than or equal to ``threshold`` receive
    weight 1.  When ``secondary`` is provided, fidelities between
    ``secondary`` and ``threshold`` are added with ``secondary_weight``.
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

class GraphQNN:
    """Convenient wrapper around the seed functions.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the MLP.
    weights : List[Tensor]
        Weight matrices for every layer, including the readout.
    """

    def __init__(self, arch: Sequence[int], weights: List[Tensor]) -> None:
        self.arch = list(arch)
        self.weights = weights

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.arch, self.weights, samples)

    def predict(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Return logits and probability vector for a single input."""
        current = features
        for weight in self.weights[:-1]:
            current = torch.tanh(weight @ current)
        logits = self.weights[-1] @ current
        probs = F.softmax(logits, dim=0)
        return logits, probs

    def train(
        self,
        dataset: List[Tuple[Tensor, Tensor]],
        loss_fn=F.cross_entropy,
        optimizer_cls=torch.optim.Adam,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Simple training loop for the MLP.

        Parameters
        ----------
        dataset : List[Tuple[Tensor, Tensor]]
            Each element is (features, target_one_hot).
        loss_fn : Callable
            Loss function that accepts logits and target indices.
        optimizer_cls : torch.optim.Optimizer
            Optimizer class to use.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        """
        # Convert targets to class indices
        targets = torch.stack([t.argmax(dim=0) for _, t in dataset])
        inputs = torch.stack([x for x, _ in dataset])

        params = [p for p in self.weights]
        optimizer = optimizer_cls(params, lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = inputs
            for weight in self.weights[:-1]:
                logits = torch.tanh(weight @ logits)
            logits = self.weights[-1] @ logits
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
