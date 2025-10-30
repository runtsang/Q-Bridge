"""Hybrid estimator combining classical PyTorch and quantum‑kernel features.

This module implements a lightweight `HybridEstimator` that can wrap any
PyTorch `nn.Module` (including a quantum kernel implemented as a
`torchquantum.QuantumModule`).  It exposes a unified `evaluate` API that
accepts an iterable of observables and a list of parameter sets.  Optional
Gaussian shot noise can be added to mimic quantum sampling noise.

The module also provides helper utilities for random network generation and
fidelity‑based graph construction, mirroring the `GraphQNN` reference.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Random weight matrix generation (GraphQNN style)
# --------------------------------------------------------------------------- #
def _random_weight_matrix(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_network(
    arch: Sequence[int], samples: int = 100
) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random feed‑forward network and training data.

    Returns:
        arch: The architecture list.
        weights: List of weight matrices.
        training_data: List of (input, target) tuples.
        target_weight: The last weight matrix (used as ground truth).
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_random_weight_matrix(in_f, out_f))
    target_weight = weights[-1]
    training_data = []
    for _ in range(samples):
        inp = torch.randn(arch[0], dtype=torch.float32)
        tgt = target_weight @ inp
        training_data.append((inp, tgt))
    return list(arch), weights, training_data, target_weight


# --------------------------------------------------------------------------- #
#  Fidelity‑based graph construction
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap between two normalized vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_n, b_n).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= threshold receive weight 1.0.
    If ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` receive ``secondary_weight``.
    """
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
#  Simple sampler and quanvolution modules (classical analogues)
# --------------------------------------------------------------------------- #
class SamplerModule(nn.Module):
    """A tiny two‑layer softmax sampler."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


def create_sampler_qnn() -> nn.Module:
    """Factory returning a sampler QNN (classical analogue)."""
    return SamplerModule()


class QuanvolutionFilter(nn.Module):
    """A classical 2‑D convolution inspired by the quanvolution example."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


def create_quanvolution_filter() -> nn.Module:
    """Factory returning a classical quanvolution filter."""
    return QuanvolutionFilter()


# --------------------------------------------------------------------------- #
#  HybridEstimator
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """Unified estimator that can wrap any PyTorch `nn.Module`.

    The model may be a pure neural network or a quantum kernel implemented as
    a `torchquantum.QuantumModule`.  The `evaluate` method accepts an iterable
    of observables (callables that map a model output to a scalar) and a list
    of parameter sets.  Optional Gaussian shot noise can be added to mimic
    quantum sampling.

    Attributes
    ----------
    model : nn.Module
        The wrapped PyTorch module.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
        parameter_sets
            Sequence of parameter vectors (each a sequence of floats).
        shots
            If provided, Gaussian noise with stddev 1/sqrt(shots) is added to
            each mean value.
        seed
            Random seed for the noise generator.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the results of all
            observables for a single parameter set.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
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
                        val = val.detach().cpu().numpy().item()
                    row.append(float(val))
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = [
    "HybridEstimator",
    "random_network",
    "fidelity_adjacency",
    "create_sampler_qnn",
    "create_quanvolution_filter",
]
