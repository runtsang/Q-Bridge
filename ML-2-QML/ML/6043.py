"""Graph-based classical neural network with message passing and fidelity‑based adjacency utilities.

This module extends the original GraphQNN by incorporating a torch_geometric
graph neural network backbone, random graph generation, and a fidelity–based
adjacency construction that can be used for downstream graph clustering or
graph‑kernel methods.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

Tensor = torch.Tensor
GraphData = Data


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


class GraphQNNGen462(nn.Module):
    """Hybrid graph neural network with fidelity‑based adjacency construction."""

    def __init__(self, arch: Sequence[int], hidden_dim: int = 32, dropout: float = 0.0):
        """
        Parameters
        ----------
        arch: Sequence[int]
            Layer sizes for a fully‑connected MLP that will process the node
            embeddings after the GCN backbone.
        hidden_dim: int, optional
            Output dimension of each GCN layer.
        dropout: float, optional
            Dropout probability applied after each GCN layer.
        """
        super().__init__()
        self.arch = list(arch)
        # Build a simple GCN backbone
        self.gcn_layers = nn.ModuleList(
            [GCNConv(in_channels=arch[0], out_channels=hidden_dim)]
        )
        for _ in range(len(arch) - 1):
            self.gcn_layers.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.dropout = dropout

        # MLP head
        mlp_layers = []
        in_dim = hidden_dim
        for out_dim in arch[1:]:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=self.dropout))
            in_dim = out_dim
        mlp_layers.pop()  # remove last dropout
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, data: GraphData) -> Tensor:
        """Propagate node features through the GCN and MLP head.

        Parameters
        ----------
        data: torch_geometric.data.Data
            Contains ``x`` (node features) and ``edge_index`` (adjacency).

        Returns
        -------
        Tensor
            Node‑level output of the network.
        """
        x, edge_index = data.x, data.edge_index
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x)

    # ------------------------------------------------------------------
    # Utility functions mirroring the original seed
    # ------------------------------------------------------------------

    @staticmethod
    def random_training_data(target_weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data for a linear target transformation."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a random linear network and training data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNNGen462.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Forward propagate through a sequence of linear layers."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Cosine‑based fidelity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphQNNGen462.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_graph(num_nodes: int, num_edges: int) -> GraphData:
        """Generate a random undirected graph with given nodes/edges."""
        G = nx.gnm_random_graph(num_nodes, num_edges)
        x = torch.randn((num_nodes, 1), dtype=torch.float32)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        # add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def generate_random_dataset(
        num_graphs: int, num_nodes: int, num_edges: int
    ) -> List[GraphData]:
        """Create a list of random graph data objects."""
        return [GraphQNNGen462.random_graph(num_nodes, num_edges) for _ in range(num_graphs)]
