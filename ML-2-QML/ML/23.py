"""GraphQNN__gen023: Classical GCN‑based implementation with a unified interface.

The original seed only provided feed‑forward utilities.  This module extends it by
adding a graph‑convolutional neural‑network (GCN) backbone, a simple training loop,
and a `GraphQNN` class that can be instantiated with a `backbone` flag to choose
between the GCN and the original point‑to‑point (p2p) linear layers.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional, Callable

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Graph = nx.Graph

# --------------------------------------------------------------------------- #
#   Helper utilities – identical to the original seed
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with ``torch.randn``."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a simple linear target dataset."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight list and a training pair set for the given architecture."""
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
    """Run a deterministic feed‑forward through the given weights."""
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
    """Return the squared overlap of two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> Graph:
    """Build a weighted graph from state correlations."""
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
#   Core GCN layers – a minimal, pure‑torch implementation
# --------------------------------------------------------------------------- #

class _GCNLayer(nn.Module):
    """Graph‑convolutional layer that multiplies node features by an adjacency matrix."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """x: (N, in_features)  ->  (N, out_features)"""
        out = torch.matmul(adj, x) @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out

# --------------------------------------------------------------------------- #
#   Unified GraphQNN class
# --------------------------------------------------------------------------- #

class GraphQNN(nn.Module):
    """Unified classical GraphQNN interface.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture list ``[in_features, hidden1, …, out_features]``.
    backbone : str
        * ``'gcn'`` (default) – use a simple GCN.
        * ``'p2p'`` – use the original point‑to‑point linear layers.
    weight_init : Callable | None
        Optional weight initializer; if None, defaults to ``torch.randn``.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        backbone: str = "gcn",
        weight_init: Optional[Callable[[int, int], Tensor]] = None,
    ):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.backbone = backbone
        self.weight_init = weight_init or _random_linear

        if backbone == "gcn":
            layers = []
            for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
                layers.append(_GCNLayer(in_f, out_f))
            self.layers = nn.ModuleList(layers)
        elif backbone == "p2p":
            self.layers = nn.ModuleList(
                [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:])]
            )
        else:
            raise ValueError(f"Unsupported backbone {backbone!r}")

    def forward(self, features: Tensor, adj: Tensor) -> Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        features : Tensor
            Node feature matrix of shape ``(N, in_features)``.
        adj : Tensor
            Normalised adjacency matrix of shape ``(N, N)``.
        """
        x = features
        for layer in self.layers:
            if isinstance(layer, _GCNLayer):
                x = layer(x, adj)
                x = F.relu(x)
            else:  # linear (p2p)
                x = torch.tanh(layer(x))
        return x

    def train_loop(
        self,
        dataset: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Simple training loop with MSE loss."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for features, target in dataset:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.forward(features.unsqueeze(0), torch.eye(features.size(0), device=device))
                loss = loss_fn(output.squeeze(0), target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {avg_loss:.6f}")

# --------------------------------------------------------------------------- #
#   Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
