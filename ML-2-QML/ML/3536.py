"""
Classical hybrid graph neural network with EstimatorQNN regression.

This module extends the original GraphQNN utilities by adding
- a lightweight fully‑connected estimator (EstimatorQNN)
- training helpers for regression tasks
- graph‑based adjacency construction from state fidelities
- an interface that can be used as a drop‑in replacement for the
  original GraphQNN class while exposing a quantum‑ready API.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
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


def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class EstimatorQNN(torch.nn.Module):
    """Simple fully‑connected regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class HybridGraphQNN:
    """Hybrid classical graph neural network that can
    1) generate a random architecture,
    2) compute feed‑forward activations,
    3) build a fidelity‑based adjacency graph,
    4) expose a lightweight EstimatorQNN regressor for training.
    """
    def __init__(self, arch: Sequence[int], device: str = "cpu") -> None:
        self.arch = list(arch)
        self.device = device
        self.weights: List[Tensor] = []
        self.training_data: List[Tuple[Tensor, Tensor]] = []
        self.target_weight: Tensor | None = None

    def random_initialize(self, samples: int = 100) -> None:
        _, self.weights, self.training_data, self.target_weight = random_network(self.arch, samples)
        self.weights = [w.to(self.device) for w in self.weights]
        self.training_data = [(x.to(self.device), y.to(self.device)) for x, y in self.training_data]

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.arch, self.weights, samples)

    def adjacency_graph(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        states = [w[0] for w in self.weights]  # use weight vectors as state proxies
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def estimator(self) -> torch.nn.Module:
        return EstimatorQNN()

    def train_estimator(self, lr: float = 1e-3, epochs: int = 200) -> torch.nn.Module:
        model = self.estimator().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            for x, y in self.training_data:
                pred = model(x.unsqueeze(0))
                loss = loss_fn(pred.squeeze(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    def predict(self, model: torch.nn.Module, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            return model(inputs.unsqueeze(0)).squeeze()


__all__ = [
    "HybridGraphQNN",
    "EstimatorQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
