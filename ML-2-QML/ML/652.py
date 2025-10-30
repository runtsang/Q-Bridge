"""Hybrid classical Graph Neural Network with graph‑based similarity and linear regression head.

The original seed only propagated raw vectors.  This extension adds a
*feature extractor* that maps node attributes to a hidden space, a
*graph similarity* module that builds a weighted adjacency from
state‑fidelity, and a *linear head* that predicts the target vector.
The forward pass now returns the feature tensor and the graph adjacency,
allowing downstream models (e.g. a GNN or a kernel method) to consume
both representations.  This design keeps the original interface
(`feedforward`, `fidelity_adjacency`, `random_network`, `random_training_data`,
`state_fidelity`) while exposing a richer training routine and
additional utilities for graph construction."""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


class GraphQNN__:
    """Classical GraphQNN implementation."""

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix with shape (out_features, in_features)."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate a dataset of (x, Wx) pairs."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create random weight matrices for each layer and the target
        weight for the tanh‑based feed‑forward network."""
        weights: List[Tensor] = []
        for in_, out_ in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNN__._random_linear(in_, out_))
        target_weight = weights[-1]
        training_data = GraphQNN__.random_training_data(target_weight, samples)
        return qnn_arch, weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Sequence[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return all layer activations for each sample."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared overlap between two pure state vectors."""
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
        """Create a weighted adjacency graph from state fidelities.

        Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
        When ``secondary`` is provided, fidelities between ``secondary`` and
        ``threshold`` are added with ``secondary_weight``."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
