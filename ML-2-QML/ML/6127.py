"""Classical graph neural network with kernel support.

Provides network generation, feedforward propagation, fidelity-based graph construction,
and RBF kernel evaluation.  The class can be instantiated with a deterministic seed
for reproducibility.  All operations are vectorised using PyTorch for performance.
"""

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import networkx as nx
import numpy as np

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    def random_network(
        self, qnn_arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(self._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = self.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(
        self, qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # Kernel utilities
    class KernalAnsatz(nn.Module):
        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    class Kernel(nn.Module):
        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.ansatz = GraphQNN.KernalAnsatz(gamma)

        def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
            x = x.view(1, -1)
            y = y.view(1, -1)
            return self.ansatz(x, y).squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
        kernel = GraphQNN.Kernel(gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["GraphQNN"]
