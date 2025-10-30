"""Hybrid classical graph neural network with training utilities.

This module extends the original GraphQNN seed by adding:
* A `GraphQNN` class that encapsulates a simple feed‑forward network with a
  *classical* embedding layer, followed by a trainable linear transformation.
* A `train` method that uses a small PyTorch `optim.Adam` loop to minimise a
  mean‑squared‑error loss between predicted and target states.
* A `graph_from_states` helper that builds a weighted adjacency graph from
  state vectors and optionally performs spectral clustering to produce a
  community label for each node.
* A `state_fidelity` helper that now accepts both tensors and numpy arrays.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Iterable as IterableType

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Array = np.ndarray


def _random_linear(
    in_features: int,
    out_features: int,
    *,
    std: float | None = None,
) -> Tensor:
    """Return a weight matrix initialized with a normal distribution."""
    std = std or 1.0 / np.sqrt(in_features)
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset of random inputs and the linear target."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        feature = torch.randn(weight.size(1), dtype=torch.float32, requires_grad=True)
        target = weight @ feature
        dataset.append((feature, target))
    return dataset


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Perform a deterministic feed‑forward pass with tanh activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor | Array, b: Tensor | Array) -> float:
    """Return the squared overlap between two state vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.abs(a_norm @ b_norm) ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor | Array],
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


class GraphQNN(nn.Module):
    """Simple feed‑forward network with an optional embedding layer."""

    def __init__(self, arch: Sequence[int], embedding_dim: int | None = None):
        super().__init__()
        self.arch = arch
        self.embedding = nn.Linear(arch[0], embedding_dim) if embedding_dim else None
        layers = []
        if embedding_dim:
            prev = embedding_dim
        else:
            prev = arch[0]
        for out in arch[1:]:
            layers.append(nn.Linear(prev, out))
            prev = out
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.embedding:
            x = self.embedding(x)
        return self.net(x)

    def train_network(
        self,
        data: Iterable[Tuple[Tensor, Tensor]],
        lr: float = 1e-3,
        epochs: int = 200,
        loss_fn=nn.MSELoss(),
    ) -> List[float]:
        """Train the network on the provided data."""
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                optimiser.zero_grad()
                pred = self(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(data))
        return losses


def graph_from_states(
    states: Sequence[Tensor | Array],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
    spectral_clusters: int | None = None,
) -> nx.Graph:
    """Create a weighted adjacency graph and optionally perform spectral clustering."""
    G = fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
    if spectral_clusters:
        # Compute Laplacian and eigenvectors for clustering
        L = nx.normalized_laplacian_matrix(G).todense()
        eigvals, eigvecs = np.linalg.eigh(L)
        # Use the first k non‑zero eigenvectors
        idx = np.argsort(eigvals)[1 : spectral_clusters + 1]
        embed = eigvecs[:, idx]
        # Assign community labels via k‑means (scikit‑learn)
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=spectral_clusters, n_init=10).fit_predict(embed)
        nx.set_node_attributes(G, {i: int(label) for i, label in enumerate(labels)}, "cluster")
    return G


__all__ = [
    "GraphQNN",
    "feedforward",
    "graph_from_states",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
