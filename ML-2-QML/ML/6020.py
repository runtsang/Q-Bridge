"""Hybrid estimator combining classical PyTorch and quantum circuit evaluation.

This module extends the original FastBaseEstimator to support graph‑based
state propagation and fidelity analysis.  It can evaluate a PyTorch model
on a batch of parameter sets, optionally adding Gaussian shot noise to
simulate measurement statistics.  The accompanying graph utilities
provide random network generation, feed‑forward activations, and a
fidelity‑based adjacency graph.
"""

from __future__ import annotations

from collections.abc import Iterable, List, Sequence
from typing import Callable, Optional

import itertools
import numpy as np
import torch
from torch import nn

import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D parameter list into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a set of parameters, with optional shot noise.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch neural network.
    shots : int | None, optional
        If provided, Gaussian noise with variance 1/shots is added to each
        output to emulate measurement shot noise.
    seed : int | None, optional
        Seed for the random number generator used for shot noise.
    """

    def __init__(self,
                 model: nn.Module,
                 *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed

    @staticmethod
    def _classical_batch(pytorch_model: nn.Module,
                         observables: Iterable[ScalarObservable],
                         parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Run a batch of PyTorch forward passes for all parameter sets."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        pytorch_model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = pytorch_model(inputs)
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

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate the model on the provided parameter sets."""
        raw = self._classical_batch(self.model, observables, parameter_sets)
        if self.shots is None:
            return raw
        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Graph‑based utilities for classical neural networks
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic training data for a linear mapping."""
    dataset: List[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and synthetic training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[torch.Tensor],
                samples: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Propagate inputs through the network and record activations."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap between two pure state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Construct a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = ["FastBaseEstimator",
           "feedforward",
           "fidelity_adjacency",
           "random_network",
           "random_training_data",
           "state_fidelity"]
