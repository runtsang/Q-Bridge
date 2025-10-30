"""Hybrid classical graph neural network utilities.

This module extends the original GraphQNN seed by adding a
HybridGraphQNN class that wraps the classical feedforward logic
and provides a training loop with a configurable loss function
and optional graph‑based regularization.  The API remains
compatible with the original functions (feedforward, fidelity_adjacency,
random_network, random_training_data, state_fidelity) so that existing
experiment scripts can be reused unchanged.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Callable

import networkx as nx
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with normal distribution, shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a list of (input, target) tuples for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        # Ensure gradients are tracked
        features = features.clone().detach().requires_grad_(True)
        target = target.clone().detach()
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create random weight matrices for each layer and a training set.

    Returns:
        arch: the network architecture.
        weights: list of weight tensors.
        training_data: list of (input, target) tuples.
        target_weight: the last layer weight matrix (ground truth).
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
    """Run a forward pass through the network for each sample."""
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
    """Return the squared overlap between two normalized vectors."""
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


def graph_regularizer(
    states: List[Tensor], graph: nx.Graph, weight: float = 1.0
) -> Tensor:
    """Compute a simple graph‑based regularization term."""
    reg = torch.tensor(0.0, device=states[0].device)
    for i, j in graph.edges():
        reg += weight * torch.norm(states[i] - states[j]) ** 2
    return reg


class HybridGraphQNN:
    """Hybrid classical graph neural network with a PyTorch backend.

    The class exposes a ``forward`` method that mirrors the original
    ``feedforward`` function and a ``train`` method that performs a
    gradient‑based optimisation of the weights.  The loss can be
    configured to include a graph‑regularization term, making it
    possible to study the effect of graph structure on learning.
    """

    def __init__(self, arch: Sequence[int], device: torch.device | None = None):
        self.arch = list(arch)
        self.device = device or torch.device("cpu")
        self.weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = torch.randn(out_f, in_f, dtype=torch.float32, requires_grad=True, device=self.device)
            self.weights.append(w)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return the activations for each layer."""
        activations = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    def predict(self, x: Tensor) -> Tensor:
        """Return the final layer activation."""
        return self.forward(x)[-1]

    def train(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 1e-3,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
        graph: nx.Graph | None = None,
        reg_weight: float = 0.0,
        verbose: bool = False,
    ) -> List[float]:
        """Simple training loop with optional graph regularizer."""
        optimizer = torch.optim.Adam(self.weights, lr=lr)
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in training_data:
                optimizer.zero_grad()
                activations = self.forward(x.to(self.device))
                pred = activations[-1]
                loss = loss_fn(pred, y.to(self.device))
                if graph is not None:
                    loss += reg_weight * graph_regularizer(activations, graph)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(training_data)
            loss_history.append(epoch_loss)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")
        return loss_history

    def get_weights(self) -> List[Tensor]:
        """Return the current weight tensors."""
        return self.weights

    def set_weights(self, weights: Sequence[Tensor]) -> None:
        """Set the network weights."""
        for w, new_w in zip(self.weights, weights):
            w.copy_(new_w)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "graph_regularizer",
    "HybridGraphQNN",
]
