"""Graph‑based hybrid neural network – Classical (PyTorch) implementation.

The API mirrors the original GraphQNN utilities while adding a SamplerQNN
module for probabilistic outputs.  All public methods are backend‑agnostic,
so the same code can be used for comparative benchmarking against the
quantum counterpart defined in the accompanying qml module."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialized weight matrix (out × in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (input, target) pairs from a linear transform."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


class GraphQuantumNeuralNetworkTorch:
    """Classical graph neural network with optional sampler integration."""

    def __init__(self, arch: Sequence[int], seed: int | None = None) -> None:
        self.arch = tuple(arch)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]
        self.target_weight = self.weights[-1]  # target of synthetic data

    def random_training_data(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Return a synthetic dataset for supervised learning."""
        return _random_training_data(self.target_weight, samples)

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Return activations for each layer for every input sample."""
        activations: List[List[Tensor]] = []
        for x, _ in samples:
            layer_vals = [x]
            current = x
            for W in self.weights:
                current = torch.tanh(W @ current)
                layer_vals.append(current)
            activations.append(layer_vals)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two unit‑normed vectors."""
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_n, b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQuantumNeuralNetworkTorch.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # Sampler integration – soft‑max output
    # ------------------------------------------------------------------
    def build_sampler(self) -> nn.Module:
        """Return a small neural sampler that consumes the last layer."""

        class SamplerModule(nn.Module):
            def __init__(self, input_dim: int, output_dim: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, output_dim * 2),
                    nn.Tanh(),
                    nn.Linear(output_dim * 2, output_dim),
                )

            def forward(self, inputs: Tensor) -> Tensor:
                return F.softmax(self.net(inputs), dim=-1)

        return SamplerModule(self.arch[-1], 2)

__all__ = [
    "GraphQuantumNeuralNetworkTorch",
    "build_sampler",
]
