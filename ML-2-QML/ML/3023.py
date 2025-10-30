"""Hybrid classical estimator that fuses PyTorch neural nets with graph‑based fidelity analysis."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import networkx as nx
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
"""Map a model output tensor to a scalar value."""

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor from a 1‑D or 2‑D iterable."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two (possibly complex) tensors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float((a_n @ b_n.conj()).item() ** 2)

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Randomly initialise a weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: torch.Tensor,
    samples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic (x, y) pairs where y = weight @ x."""
    data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        data.append((features, target))
    return data

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[Sequence[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Create a toy graph‑NN with random weights and a training set for the last layer."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_w = weights[-1]
    training = random_training_data(target_w, samples)
    return qnn_arch, weights, training, target_w

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Return layer‑wise activations for each sample."""
    outputs: List[List[torch.Tensor]] = []
    for x, _ in samples:
        activations: List[torch.Tensor] = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            activations.append(cur)
        outputs.append(activations)
    return outputs

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

class FastBaseEstimator:
    """Hybrid estimator that can evaluate a PyTorch model or a graph‑based network."""
    def __init__(
        self,
        model: nn.Module | None = None,
        qnn_arch: Sequence[int] | None = None,
        weights: List[torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.qnn_arch = qnn_arch
        self.weights = weights
        self._graph: nx.Graph | None = None

    # --------------------- Classical evaluation ------------------------------

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each parameter set using the stored PyTorch model."""
        if self.model is None:
            raise RuntimeError("No PyTorch model supplied.")
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = _ensure_batch(params)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot‑noise to the deterministic evaluation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

    # --------------------- Graph‑based evaluation ---------------------------

    def build_graph(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted adjacency graph from a list of activations."""
        self._graph = fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
        return self._graph

    def feedforward_graph(
        self,
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Propagate a dataset through the stored graph‑NN."""
        if self.qnn_arch is None or self.weights is None:
            raise RuntimeError("Graph network not configured.")
        return feedforward(self.qnn_arch, self.weights, samples)

    # --------------------- Convenience constructors ---------------------------

    @classmethod
    def from_random_network(
        cls,
        qnn_arch: Sequence[int],
        samples: int,
    ) -> "FastBaseEstimator":
        arch, weights, training, _ = random_network(qnn_arch, samples)
        return cls(model=None, qnn_arch=arch, weights=weights)

    @classmethod
    def from_random_model(
        cls,
        in_features: int,
        out_features: int,
    ) -> "FastBaseEstimator":
        weight = _random_linear(in_features, out_features)
        model = nn.Linear(in_features, out_features, bias=False)
        model.weight.data = weight
        return cls(model=model)

__all__ = ["FastBaseEstimator"]
