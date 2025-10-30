"""GraphQNN: Classical GNN with hybrid interface.

The module extends the original seed by adding a GNN encoder and a
fusion layer that can later be coupled with a quantum circuit.
All functions from the seed are preserved, with the addition of a
``GraphQNN`` class that implements a simple GCN.
"""

import itertools
from typing import Iterable, List, Tuple, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate samples for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, random weights, training data and target weight."""
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
    """Run a forward pass through the linear network."""
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
    """Squared overlap of two normalized vectors."""
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
    """Build a graph from state fidelities."""
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
    """Graph Convolutional Network for node and graph embeddings.

    Parameters
    ----------
    in_features : int
        Size of input node feature vector.
    hidden_dim : int
        Size of hidden layers.
    out_features : int
        Size of output embedding.
    num_layers : int
        Number of GCN layers.
    """

    def __init__(self, in_features: int, hidden_dim: int, out_features: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.convs: nn.ModuleList = nn.ModuleList()
        self.convs.append(nn.Linear(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, out_features)

    def forward(self, adjacency: Tensor, features: Tensor) -> Tensor:
        """Compute node embeddings."""
        h = features
        for conv in self.convs:
            h = F.relu(adjacency @ h @ conv.weight.t() + conv.bias)
        return self.out(adjacency @ h @ self.out.weight.t() + self.out.bias)

    def encode_graph(self, adjacency: Tensor, features: Tensor) -> Tensor:
        """Return a global graph embedding (mean of node embeddings)."""
        node_emb = self.forward(adjacency, features)
        return node_emb.mean(dim=0, keepdim=True)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
