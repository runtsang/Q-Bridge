# importable Python module that defines GraphQNNEstimator

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Callable, List, Tuple, Optional

import torch
import torch.nn as nn
import networkx as nx
import numpy as np

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], Tensor | float]

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialized weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create synthetic input‑output pairs for the target layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two vectors, normalised."""
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

class GraphQNNEstimator(nn.Module):
    """Classical graph‑based neural network estimator."""
    def __init__(self, arch: Sequence[int], weights: List[Tensor]):
        super().__init__()
        self.arch = list(arch)
        self.layers: nn.ModuleList = nn.ModuleList()
        for w in weights:
            linear = nn.Linear(w.size(1), w.size(0), bias=False)
            linear.weight.data = w.clone()
            self.layers.append(linear)

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int):
        """Construct a random network together with training data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    def feedforward(self, inputs: Iterable[Tensor]) -> List[List[Tensor]]:
        """Return the activations of every layer for each input."""
        activations: List[List[Tensor]] = []
        for inp in inputs:
            layer_outputs = [inp]
            current = inp
            for layer in self.layers:
                current = torch.tanh(layer(current))
                layer_outputs.append(current)
            activations.append(layer_outputs)
        return activations

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

    def add_shots(
        self,
        results: List[List[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Inject Gaussian shot noise into deterministic results."""
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = [
    "GraphQNNEstimator",
    "fidelity_adjacency",
    "state_fidelity",
    "random_training_data",
]
