"""
Classical hybrid graph‑based regression model.

This module extends the original GraphQNN utilities with a regression head
and a training loop.  It re‑uses the random graph network generation
from GraphQNN.py and the quantum‑style dataset from QuantumRegression.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence, Iterable, List, Tuple
import networkx as nx
import numpy as np

# Import utilities from the anchor GraphQNN module
from GraphQNN import (
    feedforward,
    random_network,
    fidelity_adjacency,
    state_fidelity,
    random_training_data,
)

# Import the dataset generator from the quantum regression seed
from QuantumRegression import generate_superposition_data


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset producing real‑valued features from complex quantum states."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridGraphQNNRegressor(nn.Module):
    """
    Classical regression model that first propagates inputs through a
    randomly generated graph neural network and then applies a linear
    regression head.  The graph weights are treated as learnable
    parameters, allowing the model to adapt the graph structure.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        num_features: int,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        # Build a random graph network and expose its weights as parameters
        _, weights, _, _ = random_network(qnn_arch, samples=10)
        self.weights = nn.ParameterList([nn.Parameter(w.to(self.device)) for w in weights])

        # Regression head
        self.head = nn.Linear(self.qnn_arch[-1], 1, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the graph network followed by the regression head.
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        current = x
        for weight in self.weights:
            # weight shape: (out, in)
            current = torch.tanh(weight @ current.t()).t()
        # Final activation is the output of the last layer
        out = self.head(current)
        return out.squeeze(-1)

    def train_on_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """
        One training step on a single batch.
        """
        self.train()
        self.optimizer.zero_grad()
        preds = self.forward(batch["states"])
        loss = self.loss_fn(preds, batch["target"])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, loader: Iterable[dict[str, torch.Tensor]]) -> float:
        """
        Compute mean squared error over a data loader.
        """
        self.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                preds = self.forward(batch["states"])
                loss = self.loss_fn(preds, batch["target"])
                total_loss += loss.item() * batch["states"].size(0)
                count += batch["states"].size(0)
        return total_loss / count

    def compute_fidelity_graph(self, states: torch.Tensor, threshold: float = 0.9) -> nx.Graph:
        """
        Build a graph where nodes are samples and edges are placed if the
        fidelity between their final activations exceeds *threshold*.
        """
        current = states
        for weight in self.weights:
            current = torch.tanh(weight @ current.t()).t()
        final_state = current.detach().cpu().numpy()
        return fidelity_adjacency(final_state, threshold)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(arch={self.qnn_arch}, "
            f"hidden_dim={self.hidden_dim})"
        )
