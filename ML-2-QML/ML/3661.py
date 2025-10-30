"""Combined classical graph‑based regression module.

This module fuses the graph‑QNN utilities from GraphQNN.py with the
superposition regression data generator from QuantumRegression.py.
It exposes a single entry point :class:`GraphQNNRegression` that
creates a random network, builds a regression dataset, and trains a
small feed‑forward network on the generated data.  The implementation
remains fully classical using PyTorch.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Graph‑QNN utilities  (from original GraphQNN.py)
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
#  Superposition regression data (from QuantumRegression.py)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Classical regression model
# --------------------------------------------------------------------------- #

class QModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

# --------------------------------------------------------------------------- #
#  GraphQNNRegression wrapper
# --------------------------------------------------------------------------- #

class GraphQNNRegression:
    """Unified wrapper that creates a random QNN, a regression dataset,
    and a classical feed‑forward network that learns to predict the
    target from the state vector.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        num_features: int,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
        device: torch.device | str = "cpu",
    ):
        # Build random network
        self.arch, self.weights, self.train_data, self.target_weight = random_network(
            qnn_arch, samples=100
        )

        # Compute adjacency graph from target weight activations
        self.graph = fidelity_adjacency(
            [w for w in self.weights], graph_threshold, secondary=secondary
        )

        # Build dataset
        self.dataset = RegressionDataset(samples=2000, num_features=num_features)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Build model
        self.model = QModel(num_features).to(device)
        self.device = device

    def train(self, epochs: int = 10, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                states = batch["states"].to(self.device)
                targets = batch["target"].to(self.device)
                preds = self.model(states)
                loss = criterion(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            epoch_loss /= len(self.dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(states.to(self.device)).cpu()

__all__ = [
    "GraphQNNRegression",
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
