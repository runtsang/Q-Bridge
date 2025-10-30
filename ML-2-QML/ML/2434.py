"""Hybrid classical kernel module with optional quantum augmentation.

This module defines UnifiedQuantumKernelGraph, which wraps a classical
RBF kernel and optionally a quantum kernel.  The graph utilities from
GraphQNN are also exposed for state‑based graph construction.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Classical RBF kernel
# --------------------------------------------------------------------------- #
class ClassicalKernelAnsatz(nn.Module):
    """Parameterized RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernel(nn.Module):
    """Wrapper for the RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# --------------------------------------------------------------------------- #
# 2. Quantum kernel placeholder (optional)
# --------------------------------------------------------------------------- #
class QuantumKernelPlaceholder(nn.Module):
    """Dummy quantum kernel that returns zero."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# 3. Unified kernel wrapper
# --------------------------------------------------------------------------- #
class UnifiedQuantumKernelGraph(nn.Module):
    """Hybrid kernel combining classical RBF and optional quantum kernel."""
    def __init__(self,
                 gamma: float = 1.0,
                 mix_weight: float | None = None,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.classical = ClassicalKernel(gamma)
        self.quantum = QuantumKernelPlaceholder() if use_quantum else None
        self.register_buffer("mix", torch.tensor(mix_weight if mix_weight is not None else 1.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.classical(x, y)
        if self.quantum is not None:
            out += self.mix * self.quantum(x, y)
        return out

# --------------------------------------------------------------------------- #
# 4. Kernel matrix
# --------------------------------------------------------------------------- #
def compute_kernel_matrix(a: Sequence[torch.Tensor],
                          b: Sequence[torch.Tensor],
                          gamma: float = 1.0,
                          mix_weight: float | None = None,
                          use_quantum: bool = False) -> np.ndarray:
    kernel = UnifiedQuantumKernelGraph(gamma, mix_weight, use_quantum)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 5. Graph utilities (borrowed from GraphQNN)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor,
                         samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int],
                   samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[torch.Tensor],
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
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
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
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
    "UnifiedQuantumKernelGraph",
    "compute_kernel_matrix",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
