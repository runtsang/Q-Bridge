"""GraphQNN__gen284.py – Classical GraphQNN implementation.

This module extends the original GraphQNN utilities by packaging them
into a single `GraphQNN` class that also exposes a classical sampler
and an RBF kernel.  The class follows the same public API as the
quantum counterpart, making it trivial to swap implementations
during experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Core utilities – adapted from the original GraphQNN.py
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (input, target) pairs using a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and synthetic training data."""
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
    """Propagate each sample through the network, recording activations."""
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
    """Squared overlap of two unit‑norm tensors."""
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
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Classical sampler – adapted from SamplerQNN.py
# --------------------------------------------------------------------------- #

def SamplerQNN() -> torch.nn.Module:
    """A tiny neural network that outputs a probability vector."""
    class SamplerModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 4),
                torch.nn.Tanh(),
                torch.nn.Linear(4, 2),
            )

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

# --------------------------------------------------------------------------- #
# Classical kernel – adapted from QuantumKernelMethod.py
# --------------------------------------------------------------------------- #

class KernalAnsatz(torch.nn.Module):
    """Placeholder for compatibility; implements an RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(torch.nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> torch.Tensor:
    """Compute a Gram matrix using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return torch.tensor([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Unified GraphQNN class
# --------------------------------------------------------------------------- #

class GraphQNN:
    """A container that exposes both classical and quantum‑style APIs."""
    def __init__(self, arch: Sequence[int], seed: int | None = None) -> None:
        self.arch = list(arch)
        if seed is not None:
            torch.manual_seed(seed)
        _, self.weights, _, self.target = random_network(self.arch, samples=1)

    # Feed‑forward -----------------------------------------------------------

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Propagate samples through the stored network."""
        return feedforward(self.arch, self.weights, samples)

    # Fidelity & graph -------------------------------------------------------

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a graph built from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    # Random data helpers ----------------------------------------------------

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random network and training data (classical)."""
        return random_network(arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        """Generate synthetic training data for a given weight matrix."""
        return random_training_data(weight, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute squared overlap of two tensors."""
        return state_fidelity(a, b)

    # Sampler & kernel -------------------------------------------------------

    @staticmethod
    def sampler() -> torch.nn.Module:
        """Return an instance of the classical sampler network."""
        return SamplerQNN()

    @staticmethod
    def kernel(gamma: float = 1.0) -> torch.nn.Module:
        """Return an instance of the classical RBF kernel."""
        return Kernel(gamma)

    @staticmethod
    def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> torch.Tensor:
        """Convenience wrapper for the Gram matrix."""
        return kernel_matrix(a, b, gamma)

__all__ = [
    "GraphQNN",
    "SamplerQNN",
    "Kernel",
    "KernalAnsatz",
    "kernel_matrix",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
    "feedforward",
]
