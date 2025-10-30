"""Hybrid classical graph neural network with estimator utilities.

This module merges the classical GraphQNN implementation with the lightweight
FastBaseEstimator pattern.  The network consists of a stack of linear layers
followed by tanh activations.  A FastEstimator is provided for batch evaluation
with optional Gaussian shot noise, mirroring the quantum estimator interface.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Sequence, List, Callable
import networkx as nx

Tensor = torch.Tensor
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridGraphQNN(nn.Module):
    """Classical hybrid graph neural network.

    The architecture is defined by a sequence of neuron counts per layer.
    The network is a stack of linear layers with tanh activations.
    """

    def __init__(self, architecture: Sequence[int]) -> None:
        super().__init__()
        self.architecture = list(architecture)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def feedforward(
        self, samples: Iterable[Tensor]
    ) -> List[List[Tensor]]:
        """Return the activations for each layer for a batch of samples."""
        outputs: List[List[Tensor]] = []
        for sample in samples:
            layerwise = [sample]
            current = sample
            for layer in self.layers:
                current = torch.tanh(layer(current))
                layerwise.append(current)
            outputs.append(layerwise)
        return outputs

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> tuple[Sequence[int], List[Tensor], List[tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network and synthetic training data."""
        weights = [torch.randn(out, in_) for in_, out in zip(arch[:-1], arch[1:])]
        target_weight = weights[-1]
        training_data: List[tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            feat = torch.randn(arch[0])
            tgt = target_weight @ feat
            training_data.append((feat, tgt))
        return arch, weights, training_data, target_weight

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two unit‑norm tensors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
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
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = HybridGraphQNN.state_fidelity(si, sj)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


class FastEstimator:
    """Estimator that wraps a HybridGraphQNN and adds optional shot noise."""

    def __init__(self, model: HybridGraphQNN) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables
            Callables that map the network output to a scalar or tensor.
        parameter_sets
            List of parameter vectors to feed to the network.
        shots
            If provided, Gaussian noise with variance 1/shots is added.
        seed
            Random seed for reproducibility of the noise.
        """
        obs = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32)
                if batch.ndim == 1:
                    batch = batch.unsqueeze(0)
                outputs = self.model(batch)
                row: List[float] = []
                for f in obs:
                    val = f(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = [
    "HybridGraphQNN",
    "FastEstimator",
]
