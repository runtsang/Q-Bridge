"""GraphQNNEstimator: classical graph neural network with estimator interface.

This module merges the classical GraphQNN utilities with the lightweight
FastBaseEstimator from the original project.  It provides:
- Random network generation and training data creation.
- Feed‑forward through a sequence of tanh‑activated linear layers.
- Fidelity‑based adjacency graph construction.
- A PyTorch model class that can be evaluated on batches of parameters.
- An estimator that applies arbitrary scalar observables to the model outputs.

The design keeps the original API surface (feedforward, fidelity_adjacency,
random_network, random_training_data) while adding a unified estimator
interface that mirrors the quantum side.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import torch
import networkx as nx

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], Tensor | float]

# --------------------------------------------------------------------------- #
#  Core utilities (adapted from the original GraphQNN.py)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input, target) pairs for the final weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, list of weight matrices, training data and the target weight."""
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
    """Return activations for each sample through the network."""
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
    """Squared overlap between two normalized vectors."""
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

# --------------------------------------------------------------------------- #
#  FastBaseEstimator (adapted from the original FastBaseEstimator.py)
# --------------------------------------------------------------------------- #

def _ensure_batch(values: Sequence[float]) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
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

# --------------------------------------------------------------------------- #
#  GraphQNNEstimator
# --------------------------------------------------------------------------- #

class GraphQNNEstimator(FastBaseEstimator):
    """
    A lightweight estimator that wraps a GraphQNN model.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the QNN, e.g. ``[4, 8, 2]``.
    weights : Sequence[Tensor]
        Sequence of weight matrices for each layer.
    """
    def __init__(self, qnn_arch: Sequence[int], weights: Sequence[Tensor]) -> None:
        model = self._build_model(qnn_arch, weights)
        super().__init__(model)

    @staticmethod
    def _build_model(qnn_arch: Sequence[int], weights: Sequence[Tensor]) -> torch.nn.Module:
        class _GraphQNN(torch.nn.Module):
            def __init__(self, arch, wts):
                super().__init__()
                self.arch = arch
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(in_f, out_f, bias=False) for in_f, out_f in zip(arch[:-1], arch[1:])]
                )
                # copy weights
                for layer, w in zip(self.layers, wts):
                    layer.weight.data = w
            def forward(self, x: Tensor) -> Tensor:
                out = x
                for layer in self.layers:
                    out = torch.tanh(layer(out))
                return out
        return _GraphQNN(list(qnn_arch), list(weights))

    def fidelity_adjacency(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Compute the adjacency graph of the network outputs for a dataset."""
        outputs = [self.model(sample)[0] if isinstance(sample, Tensor) else self.model(sample) for sample, _ in samples]
        return fidelity_adjacency(outputs, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "FastBaseEstimator",
    "GraphQNNEstimator",
]
