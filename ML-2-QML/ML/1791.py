"""Hybrid graph neural network trainer.

The module augments the original seed with:
* A Graph Neural Network that predicts the weight matrices of a parameter‑sharded
  quantum circuit.
* Automatic fidelity‑based graph pruning.
* A checkpoint/restore API for reproducible experiments.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Utility functions (same semantics as the seed)

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: List[int],
    weights: List[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: List[torch.Tensor],
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
# Graph QNN model
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """Hybrid classical‑quantum graph neural network.

    The network predicts a sequence of weight matrices that will be used
    to construct a parameter‑sharded quantum circuit.  Each predicted
    weight is a linear map from the previous layer size to the next.
    """

    def __init__(self, arch: List[int], hidden_dim: int = 64):
        super().__init__()
        self.arch = arch
        self.hidden_dim = hidden_dim

        # Simple message‑passing GNN: node embeddings are produced by
        # a linear layer followed by a ReLU.  We then aggregate
        # neighbour embeddings and feed them through a final linear
        # layer to produce each weight matrix.

        self.embedding = nn.Linear(arch[0], hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers = nn.ModuleList()

        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.output_layers.append(
                nn.Linear(hidden_dim, in_f * out_f)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of predicted weight matrices.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (num_nodes, feature_dim).
        edge_index : torch.Tensor
            COO index tensor of shape (2, num_edges) describing the graph.
        """
        # Node embedding
        h = F.relu(self.embedding(x))
        # Message passing (single hop)
        row, col = edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])
        h = F.relu(self.message(agg))

        # Predict weight matrices
        weights = []
        for layer, linear in enumerate(self.output_layers):
            flattened = linear(h).view(-1, self.arch[layer], self.arch[layer + 1])
            # Collapse node dimension by averaging
            w = flattened.mean(dim=0)
            weights.append(w)
        return weights

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_graph_qnn(
    model: GraphQNN,
    dataloader: DataLoader,
    target_weights: List[torch.Tensor],
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Train the GNN to predict the target weight matrices.

    The loss is a simple mean‑squared error between the predicted
    weight matrices and the ground‑truth weights.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, edge_index in dataloader:
            x = x.to(device)
            edge_index = edge_index.to(device)

            pred_weights = model(x, edge_index)

            loss = 0.0
            for pred, true in zip(pred_weights, target_weights):
                loss += criterion(pred, true.to(device))
            loss /= len(target_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} – loss: {total_loss / len(dataloader):.6f}")

# --------------------------------------------------------------------------- #
# Checkpoint API
# --------------------------------------------------------------------------- #
def save_checkpoint(model: GraphQNN, path: str | Path) -> None:
    """Persist the model state dict."""
    torch.save(model.state_dict(), Path(path))

def load_checkpoint(model: GraphQNN, path: str | Path) -> GraphQNN:
    """Load a state dict into an existing model."""
    model.load_state_dict(torch.load(Path(path), map_location="cpu"))
    return model

__all__ = [
    "GraphQNN",
    "train_graph_qnn",
    "save_checkpoint",
    "load_checkpoint",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
