"""GraphQNNGen: Classical graph neural network with estimator utilities.

This module merges the functionality of the original GraphQNN seed with
FastBaseEstimator.  The class encapsulates the network architecture,
random weight generation, feed‑forward propagation, fidelity‑based graph
construction, and a lightweight estimator that supports Gaussian shot
noise.  The API is intentionally compatible with the original seed
while adding batch evaluation and noise handling.

Typical usage:

    net = GraphQNNGen([4, 5, 3])
    arch, weights, data, target = net.random_network(samples=10)
    activations = net.feedforward(weights, data)
    graph = net.fidelity_adjacency([a[-1] for a in activations], 0.9)
    est = net.FastEstimator()
    results = est.evaluate([lambda x: x.mean(dim=-1)], [[0.1, 0.2, 0.3, 0.4]])
"""

from __future__ import annotations

import itertools
from typing import List, Tuple, Sequence, Iterable, Callable, Optional

import numpy as np
import torch
import networkx as nx
from torch import nn

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


class GraphQNNGen:
    """Classical graph‑based neural network with estimator utilities."""

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)

    def random_network(
        self, samples: int
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return self.arch, weights, training_data, target_weight

    def feedforward(
        self, weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # Estimator utilities ----------------------------------------------------
    class FastBaseEstimator:
        """Evaluate a PyTorch model for batches of inputs and observables."""

        def __init__(self, model: nn.Module) -> None:
            self.model = model

        @staticmethod
        def _ensure_batch(values: Sequence[float]) -> Tensor:
            t = torch.as_tensor(values, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t

        def evaluate(
            self,
            observables: Iterable[Callable[[Tensor], Tensor | float]],
            parameter_sets: Sequence[Sequence[float]],
        ) -> List[List[float]]:
            observables = list(observables) or [lambda out: out.mean(dim=-1)]
            results: List[List[float]] = []
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = self._ensure_batch(params)
                    outputs = self.model(inputs)
                    row: List[float] = []
                    for obs in observables:
                        val = obs(outputs)
                        if isinstance(val, Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
            return results

    class FastEstimator(FastBaseEstimator):
        """Adds Gaussian shot noise to deterministic estimates."""

        def evaluate(
            self,
            observables: Iterable[Callable[[Tensor], Tensor | float]],
            parameter_sets: Sequence[Sequence[float]],
            *,
            shots: Optional[int] = None,
            seed: Optional[int] = None,
        ) -> List[List[float]]:
            raw = super().evaluate(observables, parameter_sets)
            if shots is None:
                return raw
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in raw:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy


__all__ = ["GraphQNNGen"]
