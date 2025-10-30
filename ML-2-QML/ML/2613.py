"""Hybrid classical self‑attention module.

This module merges the linear‑time self‑attention from the original
`SelfAttention.py` seed with a graph‑based fidelity graph that can be
used to prune or re‑weight attention weights.  The class exposes a
`run` method that returns both the raw attention matrix and a
re‑weighted matrix that incorporates a similarity graph built from
the projected keys.  The graph is constructed with the same
fidelity‑adjacency routine as in the `GraphQNN.py` seed, but
operates on the projected key vectors rather than quantum states.
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
import itertools
from typing import Tuple

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a learnable weight matrix for the projection."""
    return torch.nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32))


class SelfAttentionFusion:
    """Classical self‑attention with optional graph‑based re‑weighting."""

    def __init__(self, embed_dim: int, graph_threshold: float = 0.9, graph_secondary: float | None = None):
        self.embed_dim = embed_dim
        self.query_proj = _random_linear(embed_dim, embed_dim)
        self.key_proj = _random_linear(embed_dim, embed_dim)
        self.value_proj = _random_linear(embed_dim, embed_dim)
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

    def _build_graph(self, keys: Tensor) -> nx.Graph:
        """Construct a weighted graph from key vectors using cosine similarity."""
        keys_np = keys.detach().cpu().numpy()
        graph = nx.Graph()
        graph.add_nodes_from(range(len(keys_np)))
        for (i, ki), (j, kj) in itertools.combinations(enumerate(keys_np), 2):
            fid = np.dot(ki, kj) / (np.linalg.norm(ki) * np.linalg.norm(kj) + 1e-12)
            if fid >= self.graph_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.graph_secondary is not None and fid >= self.graph_secondary:
                graph.add_edge(i, j, weight=0.5)
        return graph

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        *,
        reweight_graph: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Compute classical attention and optional graph‑based re‑weighting.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation matrix reshaped to ``embed_dim × embed_dim``; used to
            *treat* the inputs as a batch of *rotated* embeddings.
        entangle_params : np.ndarray
            Ignored in the classical sub‑module; present only to match the
            interface.
        inputs : np.ndarray shape (batch, embed_dim)
        reweight_graph : bool
            If True, re‑weight attention scores using the similarity graph.

        Returns
        -------
        Tuple[attention, reweighted] where ``reweighted`` is None if
        ``reweight_graph`` is False.
        """
        # Project inputs
        X = torch.as_tensor(inputs, dtype=torch.float32)
        Q = X @ self.query_proj.T
        K = X @ self.key_proj.T
        V = X @ self.value_proj.T

        # Compute raw attention
        scores = torch.softmax(Q @ K.T / np.sqrt(self.embed_dim), dim=-1)
        attention = scores @ V

        if not reweight_graph:
            return attention.detach().cpu().numpy(), None

        # Build graph from key projections
        graph = self._build_graph(K.detach())
        # Compute re‑weight matrix: use adjacency weights to modulate scores
        weights = np.ones_like(scores.detach().cpu().numpy())
        for i, j, data in graph.edges(data=True):
            w = data["weight"]
            weights[i, j] = w
            weights[j, i] = w
        reweighted = (scores * torch.as_tensor(weights, dtype=torch.float32)).detach().cpu().numpy()
        return attention.detach().cpu().numpy(), reweighted

__all__ = ["SelfAttentionFusion"]
