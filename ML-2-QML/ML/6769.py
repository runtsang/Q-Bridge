"""QuantumGraphEstimator – classical implementation.

This module merges the lightweight PyTorch estimator from the original
FastBaseEstimator seed with the graph‑based utilities of GraphQNN.
It exposes a unified API that can evaluate a neural network,
add shot‑noise, and build a weighted adjacency graph from the
fidelities of the outputs of each layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor for a batch of parameter vectors."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _collect_activations(model: nn.Module, inputs: torch.Tensor) -> List[torch.Tensor]:
    """Hook into linear layers to record activations."""
    activations: List[torch.Tensor] = [inputs]
    def hook(module, inp, out):
        activations.append(out.detach())
    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        _ = model(inputs)
    for h in hooks:
        h.remove()
    return activations


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of input parameters.

    The estimator accepts either an already‑created ``torch.nn.Module`` or a
    factory callable that returns one.  The model is kept immutable during
    inference, mirroring the behaviour of the original implementation.
    """

    def __init__(self, model: Union[nn.Module, Callable[[], nn.Module]]) -> None:
        self._model_factory = model

    def _model(self) -> nn.Module:
        return self._model_factory() if callable(self._model_factory) else self._model_factory

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: str | None = None,
    ) -> List[List[float]]:
        """
        Run inference on all parameter sets and return a list of rows.
        Each row contains the output of the *one‑hot* or **mean** scalar
        observables defined in the seed.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        model = self._model()
        model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                if device:
                    inputs = inputs.to(device)
                outputs = model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Wraps FastBaseEstimator to add shot‑noise simulation.

    The noise model is Gaussian with zero mean and a standard deviation
    equal to the inverse square root of the number of shots, mimicking the
    quantum measurement statistics.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, device=device)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def fidelity_graph(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from the fidelities of the outputs of each
        layer across all parameter sets.
        """
        all_states: List[torch.Tensor] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            activations = _collect_activations(self._model(), inputs)
            all_states.extend(activations)
        return fidelity_adjacency(all_states, threshold, secondary=secondary, secondary_weight=secondary_weight)


# --- Graph‑based utilities (from GraphQNN) ---------------------------------

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
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


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
