"""Hybrid estimator combining fast classical and graph-based quantum utilities.

This module unifies the lightweight PyTorch estimator, graph‑based QNN
propagation and fidelity graph construction into a single interface.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------
Tensor = torch.Tensor


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor with shape (1, N)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ----------------------------------------------------------------------
# Classical estimator
# ----------------------------------------------------------------------
class _TorchEstimator:
    """Fast deterministic estimator for PyTorch modules."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

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
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for ob in observables:
                    val = ob(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results


class _NoisyTorchEstimator(_TorchEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# ----------------------------------------------------------------------
# Graph‑based QNN utilities
# ----------------------------------------------------------------------
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
        for w in weights:
            current = torch.tanh(w @ current)
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
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----------------------------------------------------------------------
# Fully connected layer stand‑in
# ----------------------------------------------------------------------
def FCL(n_features: int = 1) -> nn.Module:
    """Return a tiny PyTorch module mimicking a quantum fully‑connected layer."""

    class _FCL(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return _FCL()


# ----------------------------------------------------------------------
# Hybrid estimator that dispatches to the appropriate backend
# ----------------------------------------------------------------------
class HybridEstimator:
    """Unified estimator that can wrap a PyTorch model, a graph‑QNN or a hybrid classifier.

    Parameters
    ----------
    model : nn.Module | tuple
        * If a PyTorch module → classical evaluation.
        * If a tuple (arch, weights, data, target) → graph‑QNN evaluation.
        * If a callable returning a tensor → hybrid classifier (fully connected).
    model_type : str, optional
        One of ``'torch'``, ``'graph'`` or ``'hybrid'``.  The default is inferred.
    """

    def __init__(self, model: Union[nn.Module, Tuple, Callable], model_type: str | None = None) -> None:
        if model_type is None:
            if isinstance(model, nn.Module):
                model_type = "torch"
            elif isinstance(model, tuple) and len(model) == 4:
                model_type = "graph"
            elif callable(model):
                model_type = "hybrid"
            else:
                raise TypeError("Could not infer model_type from the provided model.")
        self.model_type = model_type
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Dispatch evaluation to the underlying backend."""
        if self.model_type == "torch":
            estimator = _NoisyTorchEstimator(self.model)
            return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)
        elif self.model_type == "graph":
            return self._evaluate_graph(observables, parameter_sets)
        elif self.model_type == "hybrid":
            return self._evaluate_hybrid(observables, parameter_sets)
        else:
            raise ValueError(f"Unsupported model_type {self.model_type!r}")

    # ------------------------------------------------------------------
    # Backend specific helpers
    # ------------------------------------------------------------------
    def _evaluate_graph(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a graph‑QNN by feeding forward the data and applying observables."""
        arch, weights, data, _ = self.model
        # parameter_sets are indices into the data samples
        results: List[List[float]] = []
        for idx in parameter_sets:
            sample, _ = data[idx]
            activations = feedforward(arch, weights, [(sample, _)])
            row = [float(ob(activations[-1])) for ob in observables]
            results.append(row)
        return results

    def _evaluate_hybrid(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a fully‑connected hybrid layer (classical approximation)."""
        # Here we simply treat the callable as a PyTorch module with a run method.
        results: List[List[float]] = []
        for params in parameter_sets:
            out = self.model.run(params)
            row = [float(ob(torch.tensor(out))) for ob in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_torch(cls, module: nn.Module) -> "HybridEstimator":
        return cls(module, "torch")

    @classmethod
    def from_graph(cls, arch: Sequence[int], weights: Sequence[Tensor], data: Sequence[Tuple[Tensor, Tensor]]) -> "HybridEstimator":
        return cls((arch, weights, data, None), "graph")

    @classmethod
    def from_hybrid(cls, func: Callable) -> "HybridEstimator":
        return cls(func, "hybrid")


__all__ = [
    "HybridEstimator",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
    "FCL",
]
