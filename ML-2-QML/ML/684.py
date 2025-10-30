"""
GraphQNN__gen348: Classical utilities with extended graph neural‑network helpers.

Features
--------
* Keeps the original feed‑forward, adjacency, and random‑network helpers.
* Adds a PyTorch `GraphNN` module that mirrors the seed’s tanh‑based network.
* Provides a Laplacian‑embedding extractor for graph‑based node features.
* Includes a small training helper that runs a single epoch of MSE loss.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Classical utilities (unchanged from the seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic linear regression data."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and associated training data."""
    weights: List[Tensor] = []
    for in_, out_ in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_, out_))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Standard feed‑forward through a chain of tanh layers."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Compute squared‑norm‑normalized fidelity between two vectors."""
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
    """Build a graph from pairwise state fidelities."""
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
# 2.  PyTorch wrapper for the classical feed‑forward chain
# --------------------------------------------------------------------------- #
class GraphNN(nn.Module):
    """
    Simple feed‑forward neural network that mirrors the original seed's logic.
    Each layer applies a linear transform followed by tanh activation.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


# --------------------------------------------------------------------------- #
# 3.  Graph‑based feature extractor (Laplacian eigenvectors)
# --------------------------------------------------------------------------- #
def laplacian_embeddings(graph: nx.Graph, dim: int) -> torch.Tensor:
    """
    Compute the first `dim` non‑trivial Laplacian eigenvectors of `graph`.
    Returns a tensor of shape (num_nodes, dim).
    """
    L = nx.normalized_laplacian_matrix(graph).astype(float)
    eigvals, eigvecs = torch.linalg.eigh(torch.tensor(L.todense()))
    # skip the first eigenvector (all‑ones)
    return eigvecs[:, 1 : dim + 1].float()


# --------------------------------------------------------------------------- #
# 4.  Simple training helper
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    data: List[Tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.train()
    epoch_loss = 0.0
    for features, target in data:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphNN",
    "laplacian_embeddings",
    "train_one_epoch",
]
