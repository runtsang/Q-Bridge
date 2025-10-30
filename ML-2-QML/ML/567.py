"""Hybrid graph neural network with classical GNN backbone and learnable MLP head.

The module extends the original seed by adding a graph‑convolutional
feature extractor and a flexible MLP.  It keeps the original
`random_network`, `feedforward`, `state_fidelity`, and
`fidelity_adjacency` utilities, while exposing a `GraphQNN` class that
can be trained on arbitrary graph data.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Utility functions
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialized weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic training set using the target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random weight matrix for the final layer and
    a synthetic training set.  The returned architecture is
    identical to the input for compatibility.
    """
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
    """Forward propagate through the network, storing all activations."""
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
    """Return the squared inner product between two vectors."""
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
    """Build a weighted graph from state fidelities."""
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
# 2.  Classical GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """
    A hybrid graph neural network that first applies a simple
    graph‑convolutional layer to aggregate neighbor features, then
    passes the result through a flexible MLP.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input node features.
    hidden_dim : int, optional
        Size of the hidden state after the graph convolution.
    arch : Sequence[int], optional
        Architecture of the MLP head (e.g. [64, 32, 1]).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        arch: Sequence[int] | None = None,
    ):
        super().__init__()
        self.conv = nn.Linear(in_features, hidden_dim, bias=False)
        if arch is None:
            arch = [hidden_dim, 1]
        layers = []
        in_dim = hidden_dim
        for out_dim in arch:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.pop()  # remove trailing ReLU
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Parameters
        ----------
        node_features : Tensor of shape (num_nodes, in_features)
        adjacency : Tensor of shape (num_nodes, num_nodes) or sparse COO

        Returns
        -------
        Tensor of shape (num_nodes, output_dim)
        """
        # Graph convolution (simple mean aggregation)
        if adjacency.is_sparse:
            agg = torch.sparse.mm(adjacency, node_features)
        else:
            agg = adjacency @ node_features
        hidden = torch.tanh(self.conv(agg))
        out = self.mlp(hidden)
        return out

    def train_on_graph(
        self,
        graph: nx.Graph,
        features: Tensor,
        targets: Tensor,
        lr: float = 1e-3,
        epochs: int = 200,
        device: str | None = None,
    ) -> None:
        """
        Train the model on a single graph.

        Parameters
        ----------
        graph : networkx.Graph
            The graph structure.
        features : Tensor
            Node feature matrix.
        targets : Tensor
            Ground‑truth labels per node.
        lr : float, optional
            Learning rate.
        epochs : int, optional
            Number of training epochs.
        device : str | None, optional
            Device to run the training on.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        features = features.to(device)
        targets = targets.to(device)
        adjacency = nx.to_numpy_array(graph, dtype=np.float32)
        adjacency = torch.tensor(adjacency, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(features, adjacency)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
