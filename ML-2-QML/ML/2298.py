"""Graph‑based hybrid neural network – classical implementation.

This module implements a lightweight graph‑based neural network
using PyTorch.  It re‑uses the core ideas from the original
`GraphQNN.py` while adding a FastEstimator interface that
supports Gaussian shot noise, mirroring the quantum version.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List

import torch
import networkx as nx
import numpy as np

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], Tensor | float]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    """Generate synthetic input‑target pairs for a linear target."""
    dataset: List[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random feed‑forward network and associated training data."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute all layer activations for each sample."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_vals = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
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
# Estimators – deterministic and noisy batch evaluation
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate a PyTorch model on a batch of parameters."""

    def __init__(self, model: torch.nn.Module) -> None:
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
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Wrap FastBaseEstimator with optional Gaussian shot noise."""

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


# --------------------------------------------------------------------------- #
# Hybrid GraphQNN class
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """A graph‑based neural network that can be evaluated classically.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. [3, 5, 2].
    samples : int, optional
        Number of synthetic training samples to generate.
    """

    def __init__(self, arch: Sequence[int], samples: int = 1000) -> None:
        self.arch, self.weights, self.training_data, self.target_weight = random_network(
            arch, samples
        )
        self.model = torch.nn.Sequential(
            *[torch.nn.Linear(in_f, out_f, bias=False) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )
        # initialise model weights to match random_network
        with torch.no_grad():
            for w, param in zip(self.weights, self.model.parameters()):
                param.copy_(w)

    def feedforward(self, inputs: Tensor) -> List[Tensor]:
        """Return activations for a single input vector."""
        activations = [inputs]
        current = inputs
        for layer in self.model:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations

    def fidelity_graph(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Graph of state fidelities between last‑layer outputs."""
        last_layer_outputs = [act[-1] for act in feedforward(self.arch, self.weights, self.training_data)]
        return fidelity_adjacency(last_layer_outputs, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Batch evaluation with optional shot noise."""
        estimator = FastEstimator(self.model) if shots is not None else FastBaseEstimator(self.model)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = [
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
    "FastBaseEstimator",
    "FastEstimator",
    "GraphQNNHybrid",
]
