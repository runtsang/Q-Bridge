# GraphQNN__gen316.py – classical implementation

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (features, target) for a given linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical GNN architecture with corresponding weights."""
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
    """Propagate a batch of node embeddings through the GNN."""
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
    """Squared overlap between two classical feature vectors."""
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
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNHybrid(nn.Module):
    """
    Classical graph neural network that combines a convolutional encoder with
    graph‑convolution layers and fidelity‑based adjacency construction.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the GNN (including input and output dimensions).
    conv_channels : int, default 8
        Number of channels in the initial 1‑D convolutional encoder.
    node_feature_dim : int, default 1
        Dimensionality of raw node features.
    """

    def __init__(
        self,
        arch: Sequence[int],
        conv_channels: int = 8,
        node_feature_dim: int = 1,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.node_feature_dim = node_feature_dim

        # Encoder: a lightweight 1‑D CNN that projects raw node features
        self.encoder = nn.Sequential(
            nn.Conv1d(node_feature_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Graph‑convolution layers
        self.gcl = nn.ModuleList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.gcl.append(nn.Linear(in_f, out_f))
        self.norm = nn.BatchNorm1d(arch[-1])

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Node feature tensor of shape (batch, n_nodes, node_feature_dim).
        adjacency : Tensor
            Adjacency matrix of shape (n_nodes, n_nodes). Should be symmetric
            and contain 0/1 entries.

        Returns
        -------
        Tensor
            Node embeddings of shape (batch, n_nodes, arch[-1]).
        """
        batch, n_nodes, _ = x.shape
        # Encode raw features
        # Reshape to (batch, channel, n_nodes) for Conv1d
        h = self.encoder(x.transpose(1, 2))
        h = h.transpose(1, 2)  # back to (batch, n_nodes, conv_channels)

        # Propagate through graph layers
        for linear in self.gcl:
            # Aggregate neighbors
            h = torch.matmul(adjacency, h)
            h = linear(h)
            h = F.relu(h)
        h = self.norm(h)
        return h

    def compute_adjacency_from_states(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Convenience wrapper around :func:`fidelity_adjacency`."""
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
