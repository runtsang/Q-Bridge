"""Hybrid graph neural network utilities with classical and quantum interfaces.

This module extends the original GraphQNN utilities by adding estimator
capabilities inspired by FastBaseEstimator.  The class implements:

* Random network generation for a given architecture.
* Feed‑forward propagation of samples through the network.
* Construction of a fidelity‑based adjacency graph.
* Evaluation of user supplied scalar observables on the network outputs,
  with optional Gaussian shot noise to emulate measurement statistics.

The implementation is fully classical, relying on PyTorch tensors and
networkx for graph handling.  It is deliberately lightweight so that it
can be used as a drop‑in replacement for the original GraphQNN module
in the anchor repository.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], Tensor | float]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a 1‑D sequence of scalars into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class GraphQNNHybrid:
    """Hybrid classical graph‑neural‑network with estimator support."""

    def __init__(self, qnn_arch: Sequence[int], device: str | torch.device = "cpu") -> None:
        """Create an empty network that will be populated by :meth:`initialize_random_network`."""
        self.arch = list(qnn_arch)
        self.device = torch.device(device)
        self.weights: List[Tensor] | None = None

    # ------------------------------------------------------------------ #
    # Random network generation
    # ------------------------------------------------------------------ #
    def initialize_random_network(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate a random network and training data.

        Parameters
        ----------
        samples
            Number of training samples to produce.

        Returns
        -------
        training_data
            List of (input, target) pairs for the target weight.
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = _random_linear(in_f, out_f).to(self.device)
            weights.append(w)
        self.weights = weights
        target_weight = weights[-1]
        training_data = self._random_training_data(target_weight, samples)
        return training_data

    @staticmethod
    def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic input–output pairs for the target weight."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32, device=weight.device)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    # ------------------------------------------------------------------ #
    # Feed‑forward propagation
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return layer‑wise activations for each sample.

        The function applies a tanh non‑linearity after each linear
        transformation.  It is intentionally agnostic to the shape of
        ``samples``; only the input part is used.
        """
        if self.weights is None:
            raise RuntimeError("Network weights not initialised – call ``initialize_random_network`` first.")
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w in self.weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    # ------------------------------------------------------------------ #
    # Fidelity‑based graph construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute the squared inner product between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    # Estimator API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute scalar observables for each input sample.

        The method mirrors the API of the original FastBaseEstimator.
        The network is evaluated once per parameter set and the
        observables are applied to the final layer output.  Optional
        Gaussian shot noise can be added to emulate measurement
        statistics.

        Parameters
        ----------
        observables
            Callable objects that map a network output to a scalar.
        parameter_sets
            Iterable of parameter vectors that are fed to the network.
        shots
            If provided, each mean value is perturbed with normal noise
            of variance ``1/shots``.
        seed
            Random seed for the noise generator.

        Returns
        -------
        List[List[float]]
            Nested list where ``result[i][j]`` is the value of
            ``observables[j]`` for ``parameter_sets[i]``.
        """
        if self.weights is None:
            raise RuntimeError("Network weights not initialised – call ``initialize_random_network`` first.")

        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = inputs
                for w in self.weights:
                    outputs = torch.tanh(w @ outputs)
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
    # Convenience utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_random_parameters(
        qnn_arch: Sequence[int], num_samples: int, seed: int | None = None
    ) -> List[List[float]]:
        """Generate random parameter vectors for a given architecture."""
        rng = random.Random(seed)
        return [
            [rng.uniform(-np.pi, np.pi) for _ in range(f)]
            for f in qnn_arch
        ][:num_samples]

    def __repr__(self) -> str:
        return f"<GraphQNNHybrid arch={self.arch} device={self.device}>"
