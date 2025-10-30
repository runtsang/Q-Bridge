"""GraphQNNHybrid: classical implementation of a graph neural network with hybrid head and estimator.

This module extends the original GraphQNN utilities by adding a differentiable
Hybrid head that mimics a quantum expectation value, and an estimator
class that can evaluate scalar observables with optional shot‑noise.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], Tensor | float]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialised linear weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for supervised learning."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weights, synthetic training data and the target weight."""
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
    """Propagate inputs through a purely classical feed‑forward network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized tensors."""
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
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that emulates a quantum expectation value."""

    @staticmethod
    def forward(ctx, inputs: Tensor, shift: float) -> Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Simple linear head used in hybrid classification models."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class GraphQNNHybrid(nn.Module):
    """
    Unified graph‑based neural network that supports a hybrid quantum expectation head
    and fast estimator utilities.  The API mirrors the original GraphQNN while adding
    classification and estimation capabilities.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        *,
        use_hybrid_head: bool = False,
        shift: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.arch = list(qnn_arch)
        self.weights = nn.ParameterList(
            nn.Parameter(_random_linear(in_f, out_f))
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        )
        self.use_hybrid_head = use_hybrid_head
        self.shift = shift
        if use_hybrid_head:
            self.hybrid = Hybrid(self.arch[-1], shift=shift)
        else:
            self.hybrid = None

    def forward(self, x: Tensor) -> Tensor:
        """Propagate a single input through all layers."""
        for weight in self.weights:
            x = torch.tanh(weight @ x)
        if self.use_hybrid_head:
            return self.hybrid(x)
        return x

    # ------------------------------------------------------------------
    #  Utility methods that mirror the original GraphQNN API
    # ------------------------------------------------------------------
    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Return activations for a batch of samples."""
        return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Compute adjacency graph from the network's final activations."""
        final_states = [activations[-1] for activations in self.feedforward(samples=self._dummy_samples())]
        return fidelity_adjacency(final_states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def _dummy_samples(self) -> List[Tuple[Tensor, Tensor]]:
        """Generate dummy samples for graph construction."""
        return [(torch.randn(self.arch[0]), torch.zeros(self.arch[-1])) for _ in range(10)]

    # ------------------------------------------------------------------
    #  Estimator utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of scalar observables for each parameter set.
        If *shots* > 0, Gaussian shot‑noise is added to the deterministic outputs.
        """
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                output = self.forward(inputs).squeeze(0)
                row: List[float] = []
                for obs in observables:
                    val = obs(output)
                    if isinstance(val, Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results


__all__ = [
    "HybridFunction",
    "Hybrid",
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
