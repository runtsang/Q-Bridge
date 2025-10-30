from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
import networkx as nx
import numpy as np

import torch
from torch import nn

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_network(
    arch: Sequence[int],
    samples: int = 100,
    seed: int | None = None,
) -> tuple[list[int], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random fully‑connected network and a training dataset."""
    rng = np.random.default_rng(seed)
    weights: list[torch.Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(torch.tensor(rng.standard_normal((out_f, in_f)), dtype=torch.float32))
    target_weight = weights[-1]
    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(arch[0], dtype=torch.float32)
        target = target_weight @ features
        dataset.append((features, target))
    return list(arch), weights, dataset, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    inputs: torch.Tensor,
) -> list[torch.Tensor]:
    activations = [inputs]
    current = inputs
    for w in weights:
        current = torch.tanh(w @ current)
        activations.append(current)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s1), (j, s2) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s1, s2)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

class UnifiedEstimatorQNN(nn.Module):
    """
    Classical dense network that mirrors the quantum architecture for joint experiments.
    The network is initialized with random weights from `random_network`.
    It can build a fidelity graph from its hidden activations for regularization.
    """
    def __init__(self, arch: Sequence[int], seed: int | None = None, fidelity_threshold: float = 0.9) -> None:
        super().__init__()
        self.arch = list(arch)
        self.fidelity_threshold = fidelity_threshold
        self.layers = nn.ModuleList()
        rng = np.random.default_rng(seed)
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            linear = nn.Linear(in_f, out_f, bias=True)
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(rng.standard_normal((out_f, in_f)), dtype=torch.float32))
                linear.bias.zero_()
            self.layers.append(linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for layer in self.layers[:-1]:
            current = torch.tanh(layer(current))
        return self.layers[-1](current)

    def activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        act = [x]
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
            act.append(current)
        return act

    def fidelity_graph(self, activations: Sequence[torch.Tensor]) -> nx.Graph:
        return fidelity_adjacency(activations, self.fidelity_threshold)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return f"<UnifiedEstimatorQNN arch={self.arch} params={self.num_params}>"

__all__ = [
    "UnifiedEstimatorQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
