"""Enhanced classical quanvolution model combining convolution, graph‑based QNN, and estimator.

This module preserves the public API of the original Quanvolution example while
adding a graph‑based neural network (inspired by GraphQNN) and a lightweight
regression head (inspired by EstimatorQNN).  The design is intentionally
modular so that each component can be swapped or extended independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import itertools
from typing import List, Tuple, Iterable, Sequence

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Graph‑based utilities (adapted from GraphQNN)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor],
                samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Propagate a batch of samples through the graph‑QNN."""
    results: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        results.append(activations)
    return results

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two unit vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
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

# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #

class QuanvolutionHybrid(nn.Module):
    """Classical quanvolution model with a graph‑based QNN head and estimator."""

    def __init__(self,
                 conv_out_channels: int = 4,
                 graph_arch: Sequence[int] | None = None,
                 estimator_hidden: Sequence[int] | None = None) -> None:
        super().__init__()
        # Convolutional backbone (matches the original quanvolution filter)
        self.conv = nn.Conv2d(1, conv_out_channels, kernel_size=2, stride=2)

        # Graph‑based QNN
        conv_out_dim = conv_out_channels * 14 * 14
        if graph_arch is None:
            graph_arch = (conv_out_dim, 64, 4)
        self.graph_arch = list(graph_arch)
        self.graph_weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f))
             for in_f, out_f in zip(self.graph_arch[:-1], self.graph_arch[1:])]
        )

        # Estimator head (inspired by EstimatorQNN)
        if estimator_hidden is None:
            estimator_hidden = (8, 4)
        self.estimator = nn.Sequential(
            nn.Linear(self.graph_arch[-1], estimator_hidden[0]),
            nn.Tanh(),
            nn.Linear(estimator_hidden[0], estimator_hidden[1]),
            nn.Tanh(),
            nn.Linear(estimator_hidden[1], 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 1. Convolutional feature extraction
        features = self.conv(x).view(x.size(0), -1)

        # 2. Graph‑QNN propagation
        current = features
        for weight in self.graph_weights:
            current = torch.tanh(weight @ current.t()).t()

        # 3. Estimation / classification head
        output = self.estimator(current)
        return output

__all__ = ["QuanvolutionHybrid"]
