"""
GraphQNN__gen022.py

Hybrid classical‑quantum module that extends the original seed by adding:
* a PyTorch Geometric GNN that learns to predict the unitary matrix that the
  quantum network realizes.
* conversion utilities between flat tensors and unitary matrices.
* training utilities for the GNN and a simple fidelity‑based graph builder.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional PyTorch Geometric import
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None

# ---------- Utility functions ----------
def _tensor_to_unitary(tensor: torch.Tensor) -> torch.Tensor:
    """Orthogonalise a matrix to make it unitary via QR decomposition."""
    flat = tensor.flatten()
    q, _ = torch.linalg.qr(flat.unsqueeze(1))
    unitary = q.squeeze()
    return unitary.reshape(tensor.shape)

def _unitary_to_tensor(unitary: torch.Tensor) -> torch.Tensor:
    """Flatten a unitary matrix to a vector."""
    return unitary.flatten()

# ---------- Classical GNN ----------
class GNNPredictor(nn.Module):
    """Simple GCN that predicts a flattened unitary matrix."""
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required for GNNPredictor")
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Global mean pooling
        x = torch.mean(x, dim=0, keepdim=True)
        out = self.fc(x)
        return out

# ---------- Main hybrid class ----------
class GraphQNN__gen022:
    """Hybrid classical‑quantum graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Architecture of the quantum network (number of features per layer).
    depth : int, optional
        Depth of the parameterised quantum ansatz (used only for naming).
    """
    def __init__(self, arch: Sequence[int], depth: int = 2, device: str | None = None):
        self.arch = tuple(arch)
        self.depth = depth
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Classical GNN: input dimension equals first layer size
        self.gnn = GNNPredictor(
            in_channels=arch[0],
            hidden_dim=32,
            out_dim=arch[-1] * arch[0],  # flattened unitary
        ).to(self.device)

        self.opt = torch.optim.Adam(self.gnn.parameters(), lr=1e-3)

    # ------------ Random network generation ------------
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Sequence[int], Sequence[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Return architecture, weights, training data and target weight."""
        weights: list[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = GraphQNN__gen022.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(target_weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic data for training the GNN."""
        dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            dataset.append((features, target_weight))
        return dataset

    # ------------ Classical training ------------
    def train_classical(self, dataset: list[tuple[torch.Tensor, torch.Tensor]], epochs: int = 200):
        """Train the GNN to predict the target weight."""
        self.gnn.train()
        for epoch in range(epochs):
            loss = 0.0
            for features, target in dataset:
                features = features.to(self.device).unsqueeze(0)  # shape (1, in_f)
                target = target.to(self.device)  # shape (out_f, in_f)
                # Create a trivial fully‑connected graph with self‑loops
                if GCNConv is None:
                    continue
                num_nodes = features.size(0)
                if num_nodes == 1:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    edge_index = torch.tensor(list(itertools.combinations(range(num_nodes), 2)), dtype=torch.long, device=self.device).t()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                # Add self‑loops
                self_loops = torch.arange(num_nodes, dtype=torch.long, device=self.device).unsqueeze(0).repeat(2, 1)
                edge_index = torch.cat([edge_index, self_loops], dim=1)
                pred = self.gnn(features, edge_index)
                pred_unitary = _tensor_to_unitary(pred.reshape(self.arch[-1], self.arch[0]))
                loss += F.mse_loss(pred_unitary, target)
            loss /= len(dataset)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    # ------------ Feedforward ------------
    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Classical feed‑forward through the network."""
        stored: list[list[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    # ------------ Fidelity helpers ------------
    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen022.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------ GNN prediction utilities ------------
    def predict_unitary(self, graph: nx.Graph) -> torch.Tensor:
        """Predict the unitary matrix for the given graph."""
        if GCNConv is None:
            raise RuntimeError("torch_geometric is required for prediction")
        n = graph.number_of_nodes()
        # Random node features of appropriate dimension
        x = torch.randn(n, self.arch[0], dtype=torch.float32, device=self.device)
        # Edge index
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long, device=self.device).t()
        # Add self‑loops
        self_loops = torch.arange(n, device=self.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        pred = self.gnn(x, edge_index)
        unitary = _tensor_to_unitary(pred.reshape(self.arch[-1], self.arch[0]))
        return unitary

__all__ = [
    "GraphQNN__gen022",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
