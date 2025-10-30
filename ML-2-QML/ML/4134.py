"""Hybrid classifier with classical graph features and a simulated quantum layer."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor


class HybridQuantumClassifier(nn.Module):
    """
    A fully classical PyTorch model that mimics the structure of a quantum
    classifier.  It builds a fidelity‑based graph from the input batch,
    aggregates neighbours, passes the result through a small feed‑forward
    network and finally applies a linear layer that would correspond to
    expectation values from a variational circuit.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: Sequence[int] = (64, 32),
        fidelity_threshold: float = 0.9,
        secondary_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.fidelity_threshold = fidelity_threshold
        self.secondary_threshold = secondary_threshold

        # Classical graph‑based feature extractor
        self._graph_feature_extractor = nn.Sequential(
            nn.Linear(num_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        # “Quantum” linear layer – placeholder for expectation‑value outputs
        self.quantum_layer = nn.Linear(hidden_sizes[1], 2)

        # Normalisation
        self.norm = nn.BatchNorm1d(2)

    # ------------------------------------------------------------------ #
    #  Fidelity utilities
    # ------------------------------------------------------------------ #
    def _fidelity(self, a: Tensor, b: Tensor) -> float:
        """Squared overlap between two feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def _build_fidelity_graph(self, states: Sequence[Tensor]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self._fidelity(a, b)
            if fid >= self.fidelity_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary_threshold is not None and fid >= self.secondary_threshold:
                graph.add_edge(i, j, weight=0.5)
        return graph

    def _aggregate_graph_features(self, graph: nx.Graph, states: Sequence[Tensor]) -> Tensor:
        """Return a tensor of shape (batch, num_features) by averaging each node with its neighbours."""
        agg = []
        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            if neighbours:
                neigh_vec = torch.stack([states[n] for n in neighbours], dim=0).mean(dim=0)
                agg_vec = (states[node] + neigh_vec) / 2
            else:
                agg_vec = states[node]
            agg.append(agg_vec)
        return torch.stack(agg, dim=0)

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, num_features).
        """
        # Build fidelity graph on the batch
        graph = self._build_fidelity_graph(x)
        aggregated = self._aggregate_graph_features(graph, x)

        # Classical feature extraction
        features = self._graph_feature_extractor(aggregated)

        # Quantum placeholder layer
        logits = self.quantum_layer(features)

        return self.norm(logits)


__all__ = ["HybridQuantumClassifier"]
