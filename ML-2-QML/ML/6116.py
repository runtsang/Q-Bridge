"""Quantum-inspired classical classifier with extended architecture and training utilities."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def build_classifier_circuit(
    num_features: int,
    depth: int,
    activation: nn.Module = nn.ReLU,
    dropout: float = 0.0,
    num_classes: int = 2,
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum helper interface
    but enriched with optional dropout and a configurable activation function.
    Returns:
        * network: nn.Sequential classifier
        * encoding: list of feature indices
        * weight_sizes: flattened weight counts per layer
        * observables: label indices (for compatibility)
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, num_classes)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(num_classes))
    return network, encoding, weight_sizes, observables


class QuantumClassifierModel:
    """
    Wrapper around the classical feed‑forward network that emulates the quantum
    interface.  Provides a lightweight training loop and evaluation helpers
    for hybrid workflows.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        num_classes: int = 2,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, activation, dropout, num_classes
        )
        self.net.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> None:
        """
        Simple epoch‑based training loop.  X and y are assumed to be on the correct device.
        """
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Return classification accuracy on the provided data.
        """
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X.to(self.device))
            preds = logits.argmax(dim=1)
            correct = (preds == y.to(self.device)).sum().item()
            return correct / len(y)


__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]
