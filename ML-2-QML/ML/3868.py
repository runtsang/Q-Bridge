"""Hybrid classical kernel and graph utilities for advanced kernel learning."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import itertools
import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Multi‑channel RBF kernel with learnable widths
# --------------------------------------------------------------------------- #
class MultiChannelRBF(nn.Module):
    """Learnable RBF kernel per channel.

    Accepts tensors of shape (batch, channels, features) and returns a
    Gram matrix per channel with a dedicated gamma parameter that can be
    optimized during training.
    """
    def __init__(self, channels: int, gamma: float | None = None) -> None:
        super().__init__()
        if gamma is None:
            gamma = 1.0
        self.gamma = nn.Parameter(torch.full((channels,), gamma, dtype=torch.float32))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x, y: (B, C, F)
        x_exp = x[:, None, :, None]          # (B,1,C,F)
        y_exp = y[:, None, :, None]          # (B,1,C,F)
        diff = x_exp - y_exp
        sq = diff.pow(2).sum(dim=-1)         # (B,1,C)
        return torch.exp(-self.gamma * sq).squeeze(1)  # (B,C)

# --------------------------------------------------------------------------- #
# 2. Classical kernel‑regression model
# --------------------------------------------------------------------------- #
class KernelRegressionModel(nn.Module):
    """Linear regression that uses the multi‑channel RBF as feature map."""
    def __init__(self, channels: int, n_targets: int) -> None:
        super().__init__()
        self.kernel = MultiChannelRBF(channels)
        self.linear = nn.Linear(channels, n_targets, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        k = self.kernel(x, y)          # (B, C)
        return self.linear(k)          # (B, n_targets)

# --------------------------------------------------------------------------- #
# 3. Graph utilities for classical activations
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    data: List[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        data.append((features, target))
    return data

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor],
                samples: Iterable[tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        current = features
        layerwise = [current]
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
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

# --------------------------------------------------------------------------- #
# 4. Unified model that ties everything together
# --------------------------------------------------------------------------- #
class UnifiedKernelGraphModel(nn.Module):
    """Hybrid pipeline that couples a learnable RBF kernel with a graph‑based
    neural network.  It exposes methods for kernel evaluation, regression,
    random network generation, feedforward propagation, and fidelity‑based
    graph construction.
    """
    def __init__(self, channels: int, n_targets: int) -> None:
        super().__init__()
        self.kernel_regression = KernelRegressionModel(channels, n_targets)

    def compute_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        return self.kernel_regression.kernel(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.kernel_regression(x, y)

    def random_network(self, arch: Sequence[int], samples: int):
        return random_network(arch, samples)

    def feedforward(self, arch: Sequence[int], weights: Sequence[Tensor],
                    samples: Iterable[tuple[Tensor, Tensor]]):
        return feedforward(arch, weights, samples)

    def compute_fidelity_graph(self, activations: Iterable[Tensor], threshold: float,
                               *, secondary: float | None = None,
                               secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(activations, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], channels: int) -> np.ndarray:
    """Return a Gram matrix between two collections of vectors."""
    k = MultiChannelRBF(channels)
    return np.array([[k(x.unsqueeze(0), y.unsqueeze(0)).mean().item() for y in b] for x in a])

__all__ = [
    "MultiChannelRBF",
    "KernelRegressionModel",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "kernel_matrix",
    "UnifiedKernelGraphModel",
]
