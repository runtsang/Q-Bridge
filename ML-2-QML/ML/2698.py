"""Graph‑based regression model – classical implementation.

This module extends the original GraphQNN utilities with a lightweight
graph neural network that can be trained on the superposition dataset
from QuantumRegression.  It mirrors the `GraphQNN` API while adding
a regression head and training helpers.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphQNN import random_network, feedforward, fidelity_adjacency
from QuantumRegression import generate_superposition_data, RegressionDataset

Tensor = torch.Tensor
Device = torch.device


class GraphQNNRegression(nn.Module):
    """Classical graph neural network for regression.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the underlying graph neural network.
    num_features : int
        Dimensionality of the input feature vectors (from the dataset).
    device : str | torch.device, optional
        Execution device.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        num_features: int,
        device: str | Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.as_tensor([], device=device).device
        # Build a random graph‑based network
        arch, weights, _, target_weight = random_network(qnn_arch, samples=0)
        self.arch = arch
        self.weights = [w.to(self.device) for w in weights]
        self.target_weight = target_weight.to(self.device)

        # Simple MLP head on top of the last layer
        self.head = nn.Linear(arch[-1], 1)

        # Graph used for fidelity adjacency
        self.graph = None

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def build_fidelity_graph(self, states: Sequence[Tensor], threshold: float = 0.9) -> nx.Graph:
        """Construct a weighted graph from state fidelities."""
        self.graph = fidelity_adjacency(states, threshold)
        return self.graph

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Propagate features through the graph network and head."""
        activations = [features]
        current = features
        for weight in self.weights:
            current = torch.tanh(weight @ current.T).T
            activations.append(current)
        # Take the last layer activation as graph representation
        graph_repr = activations[-1]
        out = self.head(graph_repr)
        return out.squeeze(-1)

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------
    def fit(
        self,
        dataset: Iterable[dict[str, Tensor]],
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        """Simple training loop for the regression head."""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                states = batch["states"].to(self.device)
                targets = batch["target"].to(self.device)
                optimizer.zero_grad()
                preds = self.forward(states)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------
    def predict(self, states: Tensor) -> Tensor:
        """Predict on new data."""
        self.eval()
        with torch.no_grad():
            return self.forward(states.to(self.device))


__all__ = ["GraphQNNRegression"]
