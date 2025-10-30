"""Hybrid graph neural network combining classical GNN and Quanvolution-inspired convolution.

The module extends the original GraphQNN utilities with a
`GraphQuanvolutionQNN` class that incorporates a classical
convolutional filter (mimicking the quantum quanvolution) before
message passing over a graph.  The class is fully PyTorch‑based
and remains compatible with the original GraphQNN API.
"""

from __future__ import annotations

import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Sequence, Tuple

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility functions (original GraphQNN)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, y=Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight list and synthetic data for the target layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Simple forward pass through a linear chain of tanh layers."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a graph where edges are weighted by state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Classical quanvolution filter (mimicking the quantum version)
# --------------------------------------------------------------------------- #

class ClassicalQuanvolutionFilter(nn.Module):
    """A lightweight 1‑D convolution that emulates the structure of a quanvolution filter."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))

# --------------------------------------------------------------------------- #
# Hybrid graph‑neural‑network class
# --------------------------------------------------------------------------- #

class GraphQuanvolutionQNN(nn.Module):
    """
    A hybrid GNN that first applies a classical quanvolution filter to node
    features, then performs graph‑based message passing using the adjacency
    matrix derived from state fidelities.  The architecture mirrors the
    classical GraphQNN but adds a quantum‑inspired convolution step.
    """
    def __init__(
        self,
        graph: nx.Graph,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ):
        super().__init__()
        self.graph = graph
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Classical quanvolution filter
        self.qfilter = ClassicalQuanvolutionFilter(in_features, hidden_features)

        # Linear layers for message passing
        self.linear_msg = nn.Linear(hidden_features, hidden_features)
        self.linear_out = nn.Linear(hidden_features, out_features)

        # Normalised adjacency matrix (row‑normalised)
        adj = nx.to_numpy_array(graph, dtype=float)
        deg = adj.sum(axis=1, keepdims=True)
        self.register_buffer(
            "adj_norm",
            torch.tensor(adj / (deg + 1e-12), dtype=torch.float32),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape (num_nodes, in_features).

        Returns
        -------
        Tensor
            Node embeddings of shape (num_nodes, out_features).
        """
        # Apply quanvolution filter
        h = self.qfilter(x)  # (N, hidden_features)

        # Message passing: aggregate neighbour features
        h = self.adj_norm @ h  # (N, hidden_features)

        # Non‑linear transformation
        h = F.relu(self.linear_msg(h))

        # Output layer
        out = self.linear_out(h)
        return out

    # --------------------------------------------------------------------- #
    # Convenience helpers that expose the original GraphQNN API
    # --------------------------------------------------------------------- #
    def random_graph_network(self, samples: int):
        """Generate a random graph and associated random weights."""
        qnn_arch, weights, training_data, target_weight = random_network(
            [self.in_features, self.hidden_features, self.out_features], samples
        )
        return qnn_arch, weights, training_data, target_weight

    def compute_fidelity_graph(self, states: Sequence[Tensor], threshold: float):
        """Return a graph where edges are weighted by fidelity."""
        return fidelity_adjacency(states, threshold)

__all__ = [
    "GraphQuanvolutionQNN",
    "ClassicalQuanvolutionFilter",
    "feedforward",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
