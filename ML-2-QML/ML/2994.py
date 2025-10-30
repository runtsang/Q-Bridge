"""GraphQNNGen262 – Classical GNN with optional RBF kernel support.

The module mirrors the original GraphQNN utilities while adding a lightweight
kernel interface.  The public API is intentionally identical to the QML
counterpart so that downstream experiments can swap implementations
without changing the calling code.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

__all__ = [
    "GraphQNNGen262",
]


class GraphQNNGen262:
    """Hybrid classical graph neural network with optional RBF kernel support.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the virtual neural network.
    gamma : float, default=1.0
        RBF kernel width.  Ignored when ``kernel_type`` is not ``"rbf"``.
    kernel_type : str, default="rbf"
        Type of kernel to use for graph construction.  Options are ``"rbf"`` and
        ``"none"`` (default).  The quantum variant of the class implements the
        ``"quantum"`` kernel.
    """

    def __init__(self, qnn_arch: Sequence[int], gamma: float = 1.0, kernel_type: str = "rbf") -> None:
        self.qnn_arch = list(qnn_arch)
        self.gamma = gamma
        self.kernel_type = kernel_type

    # ------------------------------------------------------------------ #
    #  Data generation utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random classical network and training data.

        Returns
        -------
        arch, weights, training_data, target_weight
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNNGen262._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNNGen262.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # ------------------------------------------------------------------ #
    #  Forward propagation
    # ------------------------------------------------------------------ #
    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Compute layer‑wise activations for each sample."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    # ------------------------------------------------------------------ #
    #  Fidelity helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen262.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Kernel utilities
    # ------------------------------------------------------------------ #
    class KernalAnsatz(nn.Module):
        """Pure‑Python RBF kernel implementation."""

        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    class Kernel(nn.Module):
        """Wrapper that normalises input shapes for the RBF kernel."""

        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.ansatz = GraphQNNGen262.KernalAnsatz(gamma)

        def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
            x = x.view(1, -1)
            y = y.view(1, -1)
            return self.ansatz(x, y).squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
        kernel = GraphQNNGen262.Kernel(gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])

    def graph_from_kernel(
        self,
        data: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build an adjacency graph from a kernel matrix.

        The graph uses the kernel value as edge weight; entries below
        ``threshold`` are omitted.  ``secondary`` and ``secondary_weight``
        allow a secondary band of weights for values between the two
        thresholds.
        """
        mat = self.kernel_matrix(data, data, self.gamma)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(data)))
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                weight = mat[i, j]
                if weight >= threshold:
                    graph.add_edge(i, j, weight=weight)
                elif secondary is not None and weight >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
