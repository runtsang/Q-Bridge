"""Hybrid classical neural network that mirrors a QCNN architecture and exposes graph utilities for similarity analysis.

The module defines a single class :class:`HybridQCNNGraph` that:
* Implements a fully‑connected feed‑forward network with a feature‑map layer, several convolution‑like blocks, pooling, and a sigmoid output – a lightweight replica of the original QCNNModel.
* Provides a :func:`build_graph` helper that constructs a weighted adjacency graph from the output logits using the fidelity‑based approach from the GraphQNN code.
* Includes small helpers :func:`random_training_data` and :func:`random_network` for quick sanity checks.
* Offers :func:`state_fidelity` for tensor overlap and :func:`fidelity_adjacency` for graph construction.

Typical usage::

    from hybrid_qcnn_graph import HybridQCNNGraph
    model = HybridQCNNGraph()
    X = torch.randn(32, 8)
    logits = model(X)
    graph = model.build_graph(logits, threshold=0.9)
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import torch
from torch import nn

__all__ = [
    "HybridQCNNGraph",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
    "random_network",
    "feedforward",
]


# --------------------------------------------------------------------------- #
#   Classical network (feed‑forward + conv‑pooling emulation)
# --------------------------------------------------------------------------- #
class HybridQCNNGraph(nn.Module):
    """
    A hybrid feed‑forward network that mimics the QCNN structure.
    It contains:
    * 1 feature‑map layer (fully connected + tanh)
    * 2 convolution‑like blocks (a loop of linear + tanh, repeated for each qubit pair)
    * 3 pooling layers (the last one is a linear reduction)
    * 1 output layer (sigmoid)
    """

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(nn.Linear(feature_dim, hidden_dims[0]), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dims[1], hidden_dims[2]), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dims[2], hidden_dims[3]), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dims[3], hidden_dims[4]), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dims[4], hidden_dims[5]), nn.Tanh())
        self.head = nn.Linear(hidden_dims[5], 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    # --------------------------------------------------------------------- #
    #   Graph utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def build_graph(
        outputs: torch.Tensor,
        threshold: float = 0.9,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Construct a weighted adjacency graph from the output logits.

        Parameters
        ----------
        outputs : torch.Tensor
            2‑D tensor of shape (n_samples, n_features).
        threshold : float
            Fidelity threshold for edge weight 1.
        secondary : float | None
            Optional secondary threshold for intermediate edges.
        secondary_weight : float
            Weight assigned to secondary edges.
        """
        states = outputs.detach().cpu().numpy()
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


# --------------------------------------------------------------------------- #
#   Helper functions (state fidelity, graph construction, data generation)
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the absolute squared overlap between two tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Iterable[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate random input–target pairs using a fixed weight matrix."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Generate a random weight matrix chain and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight


def feedforward(
    qnn_arch: List[int],
    weights: List[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Compute activations for each sample through the network."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored
