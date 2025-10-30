"""GraphQNN: Classical MLP wrapper with residual connections and fidelity utilities.

This module extends the original single‑layer GNN by supporting an arbitrary depth of
linear layers and an optional residual connection between consecutive layers.  The
class exposes the same public API as the seed but includes a helper to generate
synthetic training data and a method to compute a weighted adjacency graph from
state fidelities.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class GraphQNN:
    """Depth‑aware linear network that can be used as a classical baseline for QNNs."""

    def __init__(self, architecture: Sequence[int], residual: bool = False):
        """
        Parameters
        ----------
        architecture : Sequence[int]
            Layer sizes, e.g. ``[4, 8, 2]``.
        residual : bool, optional
            If True a skip connection is added between each pair of consecutive
            layers.
        """
        self.architecture = list(architecture)
        self.residual = residual
        self.weights = self._init_weights()

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix with unit‑norm columns."""
        w = torch.randn(out_features, in_features, dtype=torch.float32)
        w = w / (w.norm(dim=0, keepdim=True) + 1e-12)
        return w

    def _init_weights(self) -> List[Tensor]:
        """Initialise weights for all hidden layers."""
        return [
            self._random_linear(in_, out)
            for in_, out in zip(self.architecture[:-1], self.architecture[1:])
        ]

    @staticmethod
    def random_training_data(
        weight: Tensor,
        samples: int,
        *,
        noise: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic regression data for a fixed linear map."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=dtype)
            target = weight @ features
            target += noise * torch.randn_like(target)
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(
        architecture: Sequence[int],
        samples: int,
        *,
        residual: bool = False,
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Convenient constructor that returns an architecture, weights, data and target."""
        weights = [
            GraphQNN._random_linear(in_, out)
            for in_, out in zip(architecture[:-1], architecture[1:])
        ]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(architecture), weights, training_data, target_weight

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        *,
        return_activations: bool = True,
    ) -> List[List[Tensor]]:
        """
        Forward pass that records activations of each layer.

        Parameters
        ----------
        samples : Iterable[Tuple[Tensor, Tensor]]
            Iterable of (input, target) pairs.  Targets are ignored.
        return_activations : bool, optional
            If False only the final layer output is returned.

        Returns
        -------
        List[List[Tensor]]
            For each sample a list of activations (including the input).
        """
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer, weight in enumerate(self.weights, 1):
                pre = current
                current = torch.tanh(weight @ current)
                if self.residual and layer < len(self.weights):
                    current = current + pre
                activations.append(current)
            stored.append(activations if return_activations else [activations[-1]])
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared cosine similarity between two vectors."""
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
        """Construct a weighted graph from pairwise fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["GraphQNN"]
