"""Hybrid classical kernel combining RBF, self‑attention and graph adjacency."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Sequence

class HybridKernel(nn.Module):
    """
    Classical hybrid kernel module.
    Applies a self‑attention transform, then evaluates a radial‑basis function
    kernel on the transformed data.  Additionally, a graph of state fidelities
    can be constructed from the attention outputs.
    """

    def __init__(self, embed_dim: int = 4, gamma: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        # Parameters for self‑attention (random for demo; learnable in practice)
        self.rotation_params = nn.Parameter(torch.randn(embed_dim * 3))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

    def _attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute classical self‑attention."""
        query = inputs @ self.rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ self.entangle_params.reshape(self.embed_dim, -1)
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value between x and y."""
        x_att = self._attention(x)
        y_att = self._attention(y)
        diff = x_att - y_att
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two lists of tensors."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def fidelity_adjacency(self, states: Sequence[torch.Tensor],
                           threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build weighted graph from cosine similarity of attention states."""
        def fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
            a_norm = a / (torch.norm(a) + 1e-12)
            b_norm = b / (torch.norm(b) + 1e-12)
            return float((a_norm @ b_norm).item() ** 2)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a_state in enumerate(states):
            for j in range(i + 1, len(states)):
                fid = fidelity(a_state, states[j])
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["HybridKernel"]
