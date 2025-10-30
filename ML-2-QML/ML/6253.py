"""GraphQNNGen058: Classical graph neural network with estimator utilities.

This module implements a lightweight feed‑forward network that mirrors the
original GraphQNN interface and a FastEstimator that can evaluate scalar
observables on batches of parameter sets.  It reuses the random network
generation and fidelity‑based graph utilities from the reference seeds
while adding a PyTorch implementation that supports gradient‑based
training and batched inference.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class GraphQNNGen058(nn.Module):
    """A lightweight classical graph‑based neural network with estimator support."""
    def __init__(self, arch: Sequence[int], seed: int | None = None) -> None:
        super().__init__()
        self.arch = list(arch)
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        layers = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weight = torch.randn(out_f, in_f, generator=rng, dtype=torch.float32)
            bias = torch.zeros(out_f, dtype=torch.float32)
            layers.append(nn.Linear(in_f, out_f, bias=True))
            layers[-1].weight.data = weight
            layers[-1].bias.data = bias
        self.model = nn.Sequential(*layers)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int, seed: int | None = None):
        """Generate a random network and training data matching the target layer."""
        rng = np.random.default_rng(seed)
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weight = torch.tensor(rng.standard_normal((out_f, in_f)), dtype=torch.float32)
            weights.append(weight)
        target_weight = weights[-1]
        training_data = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            training_data.append((features, target))
        return list(qnn_arch), weights, training_data, target_weight

    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        """Run a batch of samples through the network and return layer‑wise activations."""
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.model:
                current = torch.tanh(layer(current))
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Return the squared overlap of two normalized vectors."""
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen058.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def estimator(self, observables: Iterable[ScalarObservable]) -> FastEstimator:
        """Return a FastEstimator that evaluates the network for given parameter sets."""
        return FastEstimator(self)

__all__ = [
    "GraphQNNGen058",
    "FastBaseEstimator",
    "FastEstimator",
]
