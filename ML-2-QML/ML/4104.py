"""GraphQNN: a hybrid classical graph neural network module.

This module merges the original classical GraphQNN utilities with
additional fully‑connected layer support and a radial‑basis‑function
kernel.  The class can generate random networks, run a forward pass,
compute fidelity‑based adjacency graphs, and evaluate a classical
kernel matrix.  It is intentionally lightweight so that it can be
used as a drop‑in replacement for the original seed.

The design follows a *combination* scaling paradigm: the same API
serves both simple linear models and more expressive kernel methods,
while still exposing the core graph‑based propagation logic.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Fully‑connected layer (classical) – inspired by FCL.py
# --------------------------------------------------------------------------- #
class FullyConnectedLayer(nn.Module):
    """A tiny learnable linear module that mimics the quantum FCL example.

    The original seed exposed a ``run`` method that accepted a list of
    parameters and returned a single expectation value.  Here we keep
    that interface while internally using a PyTorch linear layer so
    gradients can be computed if desired.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean hyperbolic tangent of the linear transform."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


# --------------------------------------------------------------------------- #
#  RBF kernel (classical) – inspired by QuantumKernelMethod.py
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel as a PyTorch module.

    The implementation follows the same API as the quantum counterpart
    but operates on pure NumPy / PyTorch tensors.  The forward method
    returns a scalar kernel value.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
#  Core Graph‑QNN utilities – adapted from GraphQNN.py
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Random weight matrix with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and corresponding training data."""
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
    """Propagate a batch of inputs through the linear network."""
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
    """Squared overlap of two normalized vectors."""
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
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "FullyConnectedLayer",
    "RBFKernel",
    "kernel_matrix",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "random_training_data",
    "state_fidelity",
]
