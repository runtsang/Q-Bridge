"""Hybrid classical Graph Neural Network with embedded Sampler.

This module extends the original GraphQNN utilities by wrapping them in a
single class interface that provides both data‑generation and sampling
capabilities.  The sampler is a small neural network that outputs a
probability distribution, mirroring the structure of the quantum sampler
in the QML counterpart.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor

class GraphQNNHybrid(nn.Module):
    """Hybrid classical graph‑neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 2]``.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)

        # Core feedforward network
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        self.core = nn.Sequential(*layers)

        # Embedded sampler network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Feedforward through the core network."""
        return self.core(x)

    def sample(self, x: Tensor) -> Tensor:
        """Generate a categorical distribution from the sampler."""
        return self.sampler(x)

    @staticmethod
    def random_training_data(target: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data matching ``target``."""
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target.size(1), dtype=torch.float32)
            target_vec = target @ features
            data.append((features, target_vec))
        return data

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Create a random network and synthetic training set."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training = GraphQNNHybrid.random_training_data(target_weight, samples)
        return list(arch), weights, training, target_weight

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Store activations for a batch of samples."""
        activations: List[List[Tensor]] = []
        for x, _ in samples:
            act: List[Tensor] = [x]
            current = x
            for layer in self.core:
                current = layer(current)
                act.append(current)
            activations.append(act)
        return activations

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            a = s_i / (torch.norm(s_i) + 1e-12)
            b = s_j / (torch.norm(s_j) + 1e-12)
            fid = float((a @ b).item() ** 2)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

__all__ = [
    "GraphQNNHybrid",
]
