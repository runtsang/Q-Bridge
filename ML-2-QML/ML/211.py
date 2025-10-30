"""GraphQNN: classical GNN with a hybrid training pipeline.

This module extends the original reference by adding a small
graph‑convolutional network that learns node embeddings and a
fully‑connected output layer.  The training loop uses a mean‑squared
error loss and optimises the weights with Adam.  A simple
“fidelity‑based” evaluation metric is kept to mirror the quantum
variant.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1.  Utility helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a torch tensor of shape (out, in) with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a toy dataset that exactly reproduces the target weight."""
    torch.manual_seed(0)
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network matching the given architecture.

    Parameters
    ----------
    qnn_arch
        Sequence of layer sizes, e.g. ``[4, 8, 2]``.
    samples
        Number of training samples to generate.

    Returns
    -------
    arch
        List of layer sizes.
    weights
        List of weight tensors.
    training_data
        List of (feature, target) tuples.
    target_weight
        The weight tensor of the final layer.
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
    """Forward pass through a purely linear network.

    Parameters
    ----------
    qnn_arch
        Architecture of the network.
    weights
        Weight tensors for each layer.
    samples
        Iterable of (features, target) tuples.

    Returns
    -------
    List of activation lists for each sample.
    """
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
    """Build a weighted graph based on state fidelities.

    Parameters
    ----------
    states
        Sequence of torch tensors.
    threshold
        Fidelity threshold for weight 1.0 edges.
    secondary
        Optional secondary threshold for weighted edges.
    secondary_weight
        Weight assigned to secondary edges.
    """
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
# 2.  Graph neural network implementation
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """A simple graph‑convolutional network mirroring the classical feed‑forward.

    Parameters
    ----------
    arch
        Sequence of hidden layer sizes including input and output dimensions.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Forward pass over a graph.

        Parameters
        ----------
        x
            Node feature matrix of shape (N, D_in).
        adj
            Normalised adjacency matrix of shape (N, N).

        Returns
        -------
        Tensor
            Node embeddings of shape (N, D_out).
        """
        h = x
        for lin in self.layers:
            h = lin(adj @ h)
            h = torch.tanh(h)
        return h

# --------------------------------------------------------------------------- #
# 3.  Training utilities
# --------------------------------------------------------------------------- #
def _adjacency_matrix(graph: nx.Graph) -> Tensor:
    """Return a normalised adjacency matrix with self‑loops."""
    adj = nx.to_numpy_array(graph, dtype=float)
    adj = adj + np.eye(adj.shape[0])
    deg = np.diag(np.power(adj.sum(axis=1), -0.5))
    return torch.from_numpy(deg @ adj @ deg).float()

def train_gnn(
    model: nn.Module,
    graph: nx.Graph,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
) -> List[float]:
    """Train the GNN on a toy dataset derived from a random target weight.

    Returns a list of training losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist: List[float] = []

    n_nodes = graph.number_of_nodes()
    adj = _adjacency_matrix(graph)

    # Build a toy target weight from a reference network
    arch = [n_nodes, 16, 8, 1]
    _, weights, train_data, target_weight = random_network(arch, 1)
    target = target_weight @ torch.randn(arch[0], dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Initialise a random feature matrix
        x = torch.randn(n_nodes, arch[0], dtype=torch.float32)
        out = model(x, adj)
        loss = F.mse_loss(out, target.expand_as(out))
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
    return loss_hist

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_gnn",
]
