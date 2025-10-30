"""Pure‑classical implementation of the FraudDetectionHybrid model.

The architecture follows the graph‑QNN pattern – transaction embeddings
are turned into a fidelity‑based adjacency graph, fed through a two‑layer
feed‑forward network, and finally classified with a linear head.
The module deliberately keeps all logic on the CPU so that it can be
used as a drop‑in replacement for the quantum version during ablation
studies or when a hardware back‑end is unavailable.

Key ideas borrowed:
* Fidelity‑based graph construction (Reference 2)
* Two‑layer Tanh network with clipping (Reference 1)
"""

from __future__ import annotations

import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence, Tuple

class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection model.

    Attributes
    ----------
    feature_extractor : nn.Sequential
        Two hidden layers with Tanh activations, mirroring the
        photonic equivalent in the seed.
    classifier : nn.Linear
        Final linear head producing class logits.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int] = (128, 64),
                 n_classes: int = 2,
                 clip_bounds: Tuple[float, float] = (-5.0, 5.0)) -> None:
        super().__init__()
        self.clip_bounds = clip_bounds
        # Build a small fully‑connected network
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)

    # ------------------------------------------------------------------ #
    # Graph utilities – copied from the graph‑QNN reference
    # ------------------------------------------------------------------ #
    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared inner product of two unit‑norm vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(self,
                           states: Sequence[torch.Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph from pairwise fidelities.

        Parameters
        ----------
        states
            Iterable of state vectors (e.g. embeddings).
        threshold
            Primary fidelity threshold for an edge of weight 1.
        secondary
            Optional secondary threshold for a lighter edge.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Parameters
        ----------
        x
            Input tensor of shape (batch, input_dim).
        """
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits

# Expose public API
__all__ = ["FraudDetectionHybrid"]
