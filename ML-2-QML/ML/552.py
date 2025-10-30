"""Graph-based neural network module with PyTorch Geometric integration.

This module extends the original GraphQNN utilities by providing a
``GraphQNN`` class that encapsulates a PyTorch Geometric GCN, training
routine, and fidelity‑based graph construction.  The API mirrors the
seed but adds data‑generation, model serialization, and evaluation
hooks, enabling end‑to‑end experiments on synthetic or real graph
datasets.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    """Graph neural network that maps node features to output features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x

    @staticmethod
    def random_graph(
        num_nodes: int,
        num_edges: int,
        in_dim: int,
        out_dim: int,
    ) -> Data:
        """Generate a synthetic graph with random features and a linear target."""
        # Random adjacency
        edges = set()
        while len(edges) < num_edges:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src!= dst:
                edges.add((src, dst))
        edge_index = torch.tensor(
            list(edges | {(b, a) for a, b in edges}), dtype=torch.long
        ).t().contiguous()

        # Node features
        x = torch.randn(num_nodes, in_dim, dtype=torch.float32)

        # Target: linear transformation of features
        weight = torch.randn(out_dim, in_dim)
        y = torch.matmul(x, weight.t())

        return Data(x=x, edge_index=edge_index, y=y)

    def train_on_loader(
        self,
        loader: DataLoader,
        epochs: int = 20,
        lr: float = 0.01,
    ):
        """Simple MSE training loop."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = F.mse_loss(out, batch.y)
                loss.backward()
                optimizer.step()

    def fidelity(self, y_true: Tensor, y_pred: Tensor) -> float:
        """Normalized squared cosine similarity between predictions and targets."""
        num = torch.sum(y_true * y_pred, dim=1)
        denom = torch.norm(y_true, dim=1) * torch.norm(y_pred, dim=1)
        return torch.mean((num / (denom + 1e-12)) ** 2).item()

    @staticmethod
    def fidelity_adjacency(
        outputs: Iterable[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph where nodes are samples and edges encode
        fidelity between their output embeddings."""
        graph = nx.Graph()
        outputs = list(outputs)
        graph.add_nodes_from(range(len(outputs)))
        for (i, a), (j, b) in itertools.combinations(enumerate(outputs), 2):
            fid = GraphQNN._state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def _state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)
