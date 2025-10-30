"""Hybrid classical graph neural network that blends GNN and QCNN ideas.

The module defines :class:`GraphQNNHybrid`, a lightweight network that applies a
QCNN-inspired dense block to each node and aggregates neighbour
information via the supplied adjacency graph.  It also exposes helper
functions for generating random training data and for constructing a state‑fidelity
adjacency graph, mirroring the interface of the original GraphQNN seed.
"""

import itertools
import networkx as nx
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Sequence, List, Tuple

class QCNNModule(nn.Module):
    """QCNN‑style dense block used as a node feature extractor."""
    def __init__(self, in_features: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.net = nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))

class GraphQNNHybrid(nn.Module):
    """Hybrid GNN that wires a :class:`QCNNModule` per node and pools across edges."""
    def __init__(self, qnn_arch: Sequence[int], graph: nx.Graph) -> None:
        super().__init__()
        self.graph = graph
        self.qcnn = QCNNModule(qnn_arch[0], qnn_arch[1:-1])
        self.out = nn.Linear(qnn_arch[-2], qnn_arch[-1])
    def forward(self, node_features: Tensor) -> Tensor:
        hidden = self.qcnn(node_features)
        agg = torch.zeros_like(hidden)
        for i in self.graph.nodes:
            neigh = list(self.graph.neighbors(i))
            if neigh:
                agg[i] = hidden[neigh].mean(dim=0)
        return self.out(agg)

def random_graph_network(qnn_arch: Sequence[int], num_nodes: int, edge_prob: float, samples: int) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], torch.Tensor]:
    """Generate a random GNN architecture and synthetic training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    training_data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(num_nodes, qnn_arch[0], dtype=torch.float32)
        target = torch.tanh(target_weight @ features.T).T
        training_data.append((features, target))
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
    return qnn_arch, weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Forward‑pass through a sequence of linear layers with tanh activations."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_out = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current.T).T
            layer_out.append(current)
        activations.append(layer_out)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two unit‑norm vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float((a_n @ b_n).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

__all__ = [
    "GraphQNNHybrid",
    "QCNNModule",
    "random_graph_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
