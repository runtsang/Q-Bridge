"""Hybrid graph neural network utilities – classical implementation."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Callable

import torch
import networkx as nx
import numpy as np

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix initialized from a standard normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a list of (features, target) pairs for a linear transformation."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_classical_network(qnn_arch: Sequence[int], samples: int):
    """Create a fully‑connected network with random weights and a small training set."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = _random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward_classical(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Propagate inputs through a classical feed‑forward network."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_vals: List[Tensor] = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations


def state_fidelity_classical(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency_classical(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from pairwise classical state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity_classical(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


class FastBaseEstimator:
    """Evaluate a PyTorch model on a batch of inputs and a list of scalar observables."""
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32)
                if batch.ndim == 1:
                    batch = batch.unsqueeze(0)
                outputs = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to deterministic estimates."""
    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class EstimatorQNN(torch.nn.Module):
    """Example fully‑connected regression network used as a light‑weight estimator."""
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


__all__ = [
    "GraphQNNHybrid",
    "random_classical_network",
    "feedforward_classical",
    "state_fidelity_classical",
    "fidelity_adjacency_classical",
    "FastBaseEstimator",
    "FastEstimator",
    "EstimatorQNN",
]
