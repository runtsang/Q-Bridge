"""Combined classical graph neural network with QCNN‑style layers.

This module fuses the GraphQNN utilities with the QCNN model structure,
providing a hybrid graph neural network that aggregates node features
through learned adjacency derived from state fidelities.  The
architecture is inspired by the QCNN model but operates purely classically.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility functions – identical to the original GraphQNN but extended
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
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
# Hybrid Graph‑QCNN model
# --------------------------------------------------------------------------- #

class HybridGraphQNN(nn.Module):
    """
    A hybrid graph neural network that combines QCNN-style convolution and pooling
    with graph‑based message passing.  The architecture is inspired by the QCNN
    model but operates purely classically.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_fidelity: bool = True,
        pooling: bool = True,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.use_fidelity = use_fidelity
        self.pooling = pooling

        # QCNN‑style feature extractor
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def _graph_convolution(self, features: Tensor, adjacency: np.ndarray) -> Tensor:
        """
        Simple message‑passing: aggregate neighbour states via adjacency
        and apply a linear transformation.
        """
        adj_torch = torch.from_numpy(adjacency).float()
        return adj_torch @ features

    def forward(self, graph: nx.Graph, node_features: Tensor) -> Tensor:
        """
        Parameters
        ----------
        graph : networkx.Graph
            Graph whose adjacency defines the message‑passing pattern.
        node_features : torch.Tensor
            Tensor of shape (N, F) where N = number of nodes, F = feature dim.

        Returns
        -------
        torch.Tensor
            Output of shape (N, 1) – per‑node prediction.
        """
        # Build adjacency matrix
        adj = nx.to_numpy_array(graph, dtype=torch.float32)
        if self.use_fidelity:
            # Compute fidelity graph from current activations
            states = [node_features[i] for i in range(node_features.shape[0])]
            fid_graph = fidelity_adjacency(states, threshold=0.9)
            adj = np.maximum(adj, nx.to_numpy_array(fid_graph, dtype=np.float32))

        # Graph convolution
        out = self._graph_convolution(node_features, adj)

        # QCNN‑style layers
        out = self.feature_map(out)
        out = self.conv1(out)
        if self.pooling:
            out = self.pool1(out)
        out = self.conv2(out)
        if self.pooling:
            out = self.pool2(out)
        out = self.conv3(out)
        out = torch.sigmoid(self.head(out))
        return out

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "HybridGraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
