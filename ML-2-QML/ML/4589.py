from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Sequence, Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# Classical RBF kernel – compatible with the quantum interface
# --------------------------------------------------------------------------- #
class RBFFunction(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Hybrid Graph‑QNN – classical side
# --------------------------------------------------------------------------- #
class HybridGraphQNN(nn.Module):
    def __init__(self, arch: Sequence[int], gamma: float = 1.0):
        super().__init__()
        self.arch = list(arch)
        self.gamma = gamma
        # Random linear layers
        self.weights = nn.ParameterList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.weights.append(nn.Parameter(torch.randn(out_f, in_f)))
        self.kernel = RBFFunction(gamma)

    # --------------------------------------------------------------------- #
    # Utility: random network & training data
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.nn.Parameter], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        weights: List[torch.nn.Parameter] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.nn.Parameter(torch.randn(out_f, in_f)))
        target = weights[-1]
        training_data = HybridGraphQNN.random_training_data(target, samples)
        return list(arch), weights, training_data, target

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def feedforward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for w in self.weights:
            out = torch.tanh(w @ out)
        return out

    # --------------------------------------------------------------------- #
    # Fidelity helpers
    # --------------------------------------------------------------------- #
    def state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(self, states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, j in itertools.combinations(range(len(states)), 2):
            fid = self.state_fidelity(states[i], states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # Kernel matrix
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridGraphQNN"]
