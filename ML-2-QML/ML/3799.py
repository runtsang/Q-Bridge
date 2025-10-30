"""Unified estimator for classical neural networks with graph-based analysis.

This module extends the original FastBaseEstimator by adding:
- batched evaluation of parameter sets
- optional Gaussian shot noise
- fidelity graph construction from torch tensors
- utilities to generate random toy networks and training data
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

import networkx as nx


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
ArrayOrTensor = Union[torch.Tensor, List[float], Sequence[float]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor.unsqueeze_(0)
    return tensor


class UnifiedQNNEstimator:
    """Evaluate a PyTorch neural network for batches of parameters and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        If ``shots`` is provided, Gaussian noise with variance 1/shots is added
        to each deterministic output to mimic measurement shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------ #
    #  Graph-based fidelity utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap between two normalized tensors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_graph(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted adjacency graph constructed from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1 :], start=i + 1):
                fid = UnifiedQNNEstimator.fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Random network / training data helpers (toy)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(UnifiedQNNEstimator._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = UnifiedQNNEstimator.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored


__all__ = ["UnifiedQNNEstimator"]
