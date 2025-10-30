"""Hybrid self‑attention + graph neural network for classical data.

The module defines a single ``HybridSelfAttentionGraphQNN`` class that
combines a trainable dense‑layer stack, a multi‑head self‑attention block,
and a fidelity‑based graph builder that operates on the hidden activations.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import torch
from torch import nn

__all__ = ["HybridSelfAttentionGraphQNN"]


class HybridSelfAttentionGraphQNN(nn.Module):
    """Hybrid classical self‑attention + graph neural network."""

    def __init__(
        self,
        embed_dim: int = 4,
        num_heads: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], embed_dim),
        )
        self.hidden_states: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through attention and feed‑forward."""
        attn_out, _ = self.attn(x, x, x)
        self.hidden_states = attn_out.detach()
        return self.feedforward(attn_out)

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap of two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def compute_fidelity_graph(
        self,
        threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from hidden‑state fidelities."""
        if self.hidden_states is None:
            raise RuntimeError("Run forward() before computing a graph.")
        states = self.hidden_states.reshape(-1, self.embed_dim)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(
        weight: torch.Tensor, samples: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic (feature, target) pairs."""
        data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            data.append((features, target))
        return data

    @staticmethod
    def random_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Create a random dense network and training data."""
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f))
        target_weight = weights[-1]
        training_data = HybridSelfAttentionGraphQNN.random_training_data(
            target_weight, samples
        )
        return list(qnn_arch), weights, training_data, target_weight

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Propagate samples through a dense network."""
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from a list of state vectors."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = HybridSelfAttentionGraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
