"""Hybrid classical estimator that extends FastBaseEstimator with graph utilities and a sampler.

The estimator accepts a PyTorch ``nn.Module`` and provides:
* evaluation of observables with optional Gaussian shot noise.
* generation of random feed‑forward networks and associated training data.
* construction of a fidelity‑based adjacency graph from the network activations.
* a convenient interface to create a small sampler network for generative tasks.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable

import networkx as nx

# Local utilities from the original seed
from.FastBaseEstimator import FastBaseEstimator
from.GraphQNN import (
    random_network as classic_random_network,
    feedforward as classic_feedforward,
    fidelity_adjacency as classic_fidelity_adjacency,
)
from.SamplerQNN import SamplerQNN as ClassicSamplerQNN


class HybridEstimator(FastBaseEstimator):
    """Hybrid classical estimator that extends FastBaseEstimator with graph utilities and a sampler."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """Evaluate the network on ``parameter_sets`` and return the observable values.

        When ``shots`` is provided, Gaussian noise is added to the deterministic predictions.
        """
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [
            [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            for row in raw
        ]
        return noisy

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> tuple[list[int], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Return a random feed‑forward network and a small training set."""
        return classic_random_network(arch, samples)

    @staticmethod
    def feedforward(
        arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        """Propagate all samples through the network and collect activations."""
        return classic_feedforward(arch, weights, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        return classic_fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def sampler_network() -> nn.Module:
        """Return a small softmax sampler network."""
        return ClassicSamplerQNN()


__all__ = ["HybridEstimator"]
