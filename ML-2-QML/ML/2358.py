"""FastHybridEstimator – classical backbone for the combined estimator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D parameter list into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
#  Classical feed‑forward utilities (inspired by GraphQNN.py)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[tuple[torch.Tensor, torch.Tensor]] = []
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
) -> "nx.Graph":
    import networkx as nx
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
#  Hybrid estimator class
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """Base estimator that evaluates a PyTorch model for many parameter sets."""

    def __init__(self, model: nn.Module, *, dropout_prob: float | None = None) -> None:
        self.model = model
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else None

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.model(inputs)
        if self.dropout:
            out = self.dropout(out)
        return out

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        cache: bool = False,
    ) -> List[List[float]]:
        """Return a list of lists containing the mean value of each observable.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the torch.Tensor output of the model.
        parameter_sets : sequence of parameter vectors
            Each vector is a flat list of floats.
        cache : bool
            When True, identical parameter vectors are evaluated only once.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        seen: dict[tuple[float,...], List[float]] = {}
        with torch.no_grad():
            for params in parameter_sets:
                key = tuple(params)
                if cache and key in seen:
                    results.append(seen[key])
                    continue
                inputs = _ensure_batch(params)
                out = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                if cache:
                    seen[key] = row
                results.append(row)
        return results


class FastHybridEstimatorWithNoise(FastHybridEstimator):
    """Extends FastHybridEstimator with shot‑noise emulation."""

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


__all__ = ["FastHybridEstimator", "FastHybridEstimatorWithNoise", "random_network", "random_training_data", "feedforward", "state_fidelity", "fidelity_adjacency"]
