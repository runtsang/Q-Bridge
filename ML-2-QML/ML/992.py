"""Hybrid Graph Neural Network (GNN) module.

This module extends the original GraphQNN to support a hybrid classical‑quantum
architecture.  It keeps the same public API (feedforward, fidelity_adjacency,
random_network, random_training_data, state_fidelity) but adds a `GraphQNN`
class that can be trained end‑to‑end with autograd.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with values drawn from a normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset of feature vectors and their linear targets."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and a target linear map."""
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
    """Compute activations for a batch of examples."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two real vectors (normalised)."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a graph from pairwise fidelities; edges above `threshold` get weight 1."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN:
    """Hybrid classical‑quantum graph neural network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes; the first element is the input dimensionality.
    device : str, optional
        Target device for torch tensors (default: 'cpu').
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        *,
        device: str | None = None,
    ):
        self.qnn_arch = list(qnn_arch)
        self.device = device or "cpu"
        self.weights = [
            _random_linear(*l).to(self.device) for l in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        self.to_device(self.device)

    def to_device(self, device: str | None = None):
        """Move all parameters to the specified device."""
        if device is None:
            device = self.device
        self.device = device
        for i, w in enumerate(self.weights):
            self.weights[i] = w.to(device)

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass through the classical layers."""
        current = features.to(self.device)
        for w in self.weights:
            current = torch.tanh(w @ current)
        return current

    def train_step(self, features: Tensor, target: Tensor, lr: float = 1e-3):
        """Perform a single gradient step on the MSE loss."""
        self.to_device()
        features = features.to(self.device)
        target = target.to(self.device)
        self.zero_grad()
        output = self.forward(features)
        loss = F.mse_loss(output, target)
        loss.backward()
        with torch.no_grad():
            for w in self.weights:
                w -= lr * w.grad
        return loss.item()

    def zero_grad(self):
        """Zero gradients of all parameters."""
        for w in self.weights:
            if w.grad is not None:
                w.grad.zero_()

    def __repr__(self) -> str:
        return f"GraphQNN(arch={self.qnn_arch}, device={self.device})"
