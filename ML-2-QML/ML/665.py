"""GraphQNNHybrid: Classical GNN that learns to output a quantum unitary.

This module extends the seed by replacing the simple linear weight chain with
a small neural network that takes a graph adjacency matrix as input and
produces a unitary matrix.  The network is trained with MSE loss between
the predicted and target unitary.  The `train` method runs a standard
gradient‑descent loop, while `predict` returns the final unitary as a
NumPy array.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (x, y) where x is a random graph‑feature vector
    and y is the target weight matrix.  The graph features are created by
    random graph adjacency matrices with *sparsity*‐to‑sparsity‑??.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random target unitary and a training set of adjacency matrices.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        ``[num_nodes, out_dim]`` where ``num_nodes`` is the number of nodes in
        the input graph and ``out_dim`` is the dimension of the target unitary.
    samples : int
        Number of training samples to generate.
    """
    num_nodes, out_dim = qnn_arch
    # random unitary via QR decomposition
    mat = torch.randn(out_dim, out_dim, dtype=torch.complex64)
    q, r = torch.qr(mat)
    diag = torch.diagonal(r)
    unitary = q @ torch.diag(diag / torch.abs(diag))
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        G = nx.erdos_renyi_graph(num_nodes, 0.5)
        adj = nx.to_numpy_array(G, dtype=torch.float32)
        adj = torch.tensor(adj)
        dataset.append((adj, unitary))
    return dataset, unitary

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

class GraphQNNHybrid(nn.Module):
    """A tiny GNN that maps a graph adjacency matrix to a unitary matrix.

    The network consists of a single hidden linear layer followed by a
    linear output that is reshaped to a matrix and orthogonalised via QR
    decomposition to enforce unitarity.
    """

    def __init__(self, num_nodes: int, out_dim: int, hidden_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_nodes * num_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim * out_dim)
        self.out_dim = out_dim
        self.to(device)

    def forward(self, adjacency: Tensor) -> Tensor:
        """Return a unitary matrix of shape ``(out_dim, out_dim)``."""
        x = adjacency.flatten().to(self.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        unitary = x.reshape(self.out_dim, self.out_dim)
        # Enforce unitarity with QR
        q, r = torch.qr(unitary)
        diag = torch.diagonal(r)
        phase = diag / torch.abs(diag)
        unitary = q @ torch.diag(phase)
        return unitary.cpu()

    def train_model(self, dataset: List[Tuple[Tensor, Tensor]], epochs: int = 10, lr: float = 1e-3):
        """Standard MSE training loop."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            total_loss = 0.0
            for adj, target in dataset:
                pred = self.forward(adj)
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} loss: {total_loss/len(dataset):.6f}")

    def predict(self, adjacency: Tensor) -> Tensor:
        """Return the predicted unitary as a NumPy array."""
        return self.forward(adjacency).numpy()

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "fidelity_adjacency",
    "state_fidelity",
]
