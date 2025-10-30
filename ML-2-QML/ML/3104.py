"""Hybrid classical‑quantum kernel and graph neural network utilities.

This module merges the classical RBF kernel and graph‑based neural network
from the original `QuantumKernelMethod.py` and `GraphQNN.py` seeds.  It
provides:

* `KernalAnsatz` – a learnable RBF ansatz with a trainable bandwidth
  `gamma`.
* `Kernel` – a thin wrapper exposing a scalar kernel value.
* `kernel_matrix` – efficient Gram‑matrix computation for sequences of
  tensors.
* `RandomNetwork` utilities for generating toy feed‑forward networks.
* `GraphQNN` – a torch‑based graph neural network that can be used to
  propagate activations and build a fidelity‑based adjacency graph.
* `HybridKernelGraphModel` – a convenience container that bundles a
  kernel and a graph QNN.

The API retains the original names for backward compatibility while
adding batch support, learnable hyper‑parameters, and a unified
interface for hybrid experiments.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

# --------------------------------------------------------------------------- #
#  Classical RBF kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Learnable Radial Basis Function ansatz.

    Parameters
    ----------
    gamma : float, optional
        Initial bandwidth of the RBF.  Stored as a trainable
        parameter so that it can be tuned during optimisation.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``exp(-γ‖x−y‖²)`` for each pair in the batch."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Thin wrapper that exposes a scalar kernel value."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two collections of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Graph‑based neural network utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target   = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a toy feed‑forward network and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Propagate a batch of samples through a fully‑connected network."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑normed vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
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

class GraphQNN(nn.Module):
    """Graph neural network that propagates activations and builds a graph."""
    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)
        self.weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f))
             for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations for each layer."""
        activations = [x]
        current = x
        for weight in self.weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations

    def build_adjacency(self, states: Sequence[torch.Tensor], threshold: float,
                        *, secondary: float | None = None,
                        secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

# --------------------------------------------------------------------------- #
#  Hybrid container
# --------------------------------------------------------------------------- #
class HybridKernelGraphModel:
    """Convenience wrapper that bundles a kernel and a graph QNN."""
    def __init__(self, kernel_gamma: float = 1.0, graph_arch: Sequence[int] | None = None):
        self.kernel = Kernel(kernel_gamma)
        self.graph_qnn = GraphQNN(graph_arch) if graph_arch is not None else None

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.kernel.ansatz.gamma.item())

    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        if self.graph_qnn is None:
            raise RuntimeError("Graph QNN not initialised.")
        # Use the helper feedforward defined above
        return feedforward(self.graph_qnn.arch, [w for w in self.graph_qnn.weights], samples)

    def build_graph(self, states: Sequence[torch.Tensor], threshold: float,
                    *, secondary: float | None = None,
                    secondary_weight: float = 0.5) -> nx.Graph:
        if self.graph_qnn is None:
            raise RuntimeError("Graph QNN not initialised.")
        return self.graph_qnn.build_adjacency(states, threshold,
                                              secondary=secondary,
                                              secondary_weight=secondary_weight)

__all__ = [
    "KernalAnsatz", "Kernel", "kernel_matrix",
    "feedforward", "fidelity_adjacency",
    "random_network", "random_training_data",
    "state_fidelity",
    "GraphQNN",
    "HybridKernelGraphModel",
]
