"""Hybrid classical‑quantum graph neural network utilities.

This module extends the original GraphQNN.py by adding:
* A simple GCN that maps node features to a latent vector.
* A quantum variational circuit that receives the latent vector as a control register.
* A joint optimisation routine that minimises a fidelity loss between the output state and a target unitary.
* A lightweight graph‑based evaluation that re‑uses the fidelity‑based adjacency construction.

The code is intentionally lightweight and fully importable; it can be dropped into a research
pipeline that needs a quick prototype of a hybrid GNN‑QNN system.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum helpers
from.qml_code import (
    quantum_variational,
    fidelity_loss as q_fidelity_loss,
    random_quantum_unitary,
    random_quantum_training_data,
)

Tensor = torch.Tensor
QState = torch.Tensor  # statevector in flattened form


# --------------------------------------------------------------------------- #
# 1. Classical GNN
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    """Simple graph convolutional layer with ReLU activation."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # x: [N, in_features]
        # adj: [N, N] adjacency matrix (unnormalised)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return F.relu(norm_adj @ x @ self.weight.T)


class GCN(nn.Module):
    """Stacked GCN layers."""
    def __init__(self, layer_sizes: Sequence[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(GCNLayer(in_f, out_f))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return x  # latent representation


# --------------------------------------------------------------------------- #
# 2. Hybrid model
# --------------------------------------------------------------------------- #
class HybridGraphQNN(nn.Module):
    """Hybrid GNN + quantum variational circuit."""
    def __init__(self, gcn_arch: Sequence[int], q_params: int, target_unitary: Tensor):
        super().__init__()
        self.gcn = GCN(gcn_arch)
        self.q_params = nn.Parameter(torch.randn(q_params))
        self.target_unitary = target_unitary  # used for loss

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # Classical latent
        latent = self.gcn(x, adj)  # shape [N, d]
        # Pool over nodes to obtain a single vector
        latent_vec = latent.mean(dim=0)  # shape [d]
        # Quantum circuit
        state = quantum_variational(latent_vec, self.q_params)
        return state  # quantum statevector

    def loss(self, output_state: Tensor) -> Tensor:
        return q_fidelity_loss(output_state, self.target_unitary)


# --------------------------------------------------------------------------- #
# 3. Training utilities
# --------------------------------------------------------------------------- #
def train_hybrid(
    model: HybridGraphQNN,
    dataset: Iterable[Tuple[Tensor, Tensor, Tensor]],
    lr: float = 1e-3,
    epochs: int = 100,
    device: str = "cpu",
) -> List[float]:
    """Simple training loop for the hybrid model.

    Each dataset entry must be a tuple (features, adjacency, target_state).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, adj, target in dataset:
            features = features.to(device)
            adj = adj.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(features, adj)
            loss = model.loss(output)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f}")
    return losses


# --------------------------------------------------------------------------- #
# 4. Graph‑based evaluation
# --------------------------------------------------------------------------- #
def fidelity_adjacency(
    states: Sequence[Tensor],
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = q_fidelity_loss(state_i, state_j).item()
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "GCNLayer",
    "GCN",
    "HybridGraphQNN",
    "train_hybrid",
    "fidelity_adjacency",
    "random_quantum_unitary",
    "random_quantum_training_data",
]
