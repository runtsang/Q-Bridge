"""GraphQNNAdvanced – classical graph neural network with optional GCN and projection.

The class encapsulates a lightweight feed‑forward network that can be
composed of:
* A GCN layer (built on PyTorch Geometric) that aggregates node features.
* An optional linear projection to match the dimensionality of the quantum circuit.
* A classical output layer that returns the state vector for fidelity comparison.

The module is intentionally lightweight so it can be dropped into any
PyTorch training pipeline while still providing the same
``feedforward`` and ``fidelity_adjacency`` helpers as the seed.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility: GCN layer wrapper
# --------------------------------------------------------------------------- #
class _GCNBlock(nn.Module):
    """A single GCN layer followed by ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return F.relu(self.conv(x, edge_index))

# --------------------------------------------------------------------------- #
# Core utility functions (mirroring the seed)
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
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
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
# GraphQNNAdvanced class
# --------------------------------------------------------------------------- #
class GraphQNNAdvanced(nn.Module):
    """Graph‑based neural network with optional GCN and projection layer.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture describing the dimensionality of each layer.
    use_gcn : bool, default True
        If True, the first layer is a GCNConv that aggregates node features
        according to the supplied edge_index.
    projection : bool, default False
        If True, a final linear layer projects the output to the dimensionality
        of the last quantum layer, facilitating fidelity comparison.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_gcn: bool = True,
        projection: bool = False,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.use_gcn = use_gcn
        self.projection = projection

        layers: List[nn.Module] = []

        # GCN layer (if requested)
        if self.use_gcn:
            layers.append(_GCNBlock(self.qnn_arch[0], self.qnn_arch[1]))
            input_dim = self.qnn_arch[1]
            start_idx = 2
        else:
            input_dim = self.qnn_arch[0]
            start_idx = 1

        # Remaining linear layers
        for idx in range(start_idx, len(self.qnn_arch)):
            layers.append(nn.Linear(input_dim, self.qnn_arch[idx]))
            input_dim = self.qnn_arch[idx]

        # Optional projection to match quantum output dimension
        if self.projection:
            layers.append(nn.Linear(self.qnn_arch[-1], self.qnn_arch[-1]))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        """Return the final output state vector.

        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape (num_nodes, in_features).
        edge_index : Tensor, optional
            Edge index tensor for the GCN layer. Required if ``use_gcn`` is True.
        """
        out = x
        for layer in self.layers:
            if isinstance(layer, _GCNBlock):
                if edge_index is None:
                    raise ValueError("edge_index must be provided when use_gcn is True")
                out = layer(out, edge_index)
            else:
                out = torch.tanh(layer(out))
        return out

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Convenience wrapper that mirrors the seed implementation."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        return random_training_data(weight, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ):
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "GraphQNNAdvanced",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
