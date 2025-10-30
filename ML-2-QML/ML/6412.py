"""
QuantumClassifierModel – Classical implementation.

This variant mirrors the original QuantumClassifierModel API while
integrating the EstimatorQNN-inspired architecture.  It supports
classification and regression, shares a common `task` flag, and
provides a placeholder for quantum prediction that can be swapped
out with a real quantum backend.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """Hybrid model with a classical feed‑forward backbone.

    The class shares a common interface with its quantum counterpart.
    It can be trained with a simple supervised loop and exposes a
    ``quantum_predict`` method that can be overridden by a quantum
    implementation.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        task: str = "classification",
        device: str = "cpu",
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.task = task
        self.device = device

        # Build a deep feed‑forward backbone
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features

        if task == "classification":
            layers.append(nn.Linear(in_dim, 2))
        elif task == "regression":
            layers.append(nn.Linear(in_dim, 1))
        else:
            raise ValueError("task must be 'classification' or'regression'")

        self.net = nn.Sequential(*layers).to(device)

        # Placeholders for quantum parameters (to be filled by a QML
        # implementation if desired)
        self.input_params: list[str] = []
        self.weight_params: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))

    def parameters(self):
        return self.net.parameters()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Simple supervised training loop."""
        X, y = X.to(self.device), y.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = (
            nn.CrossEntropyLoss()
            if self.task == "classification"
            else nn.MSELoss()
        )

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    def quantum_predict(self, X: torch.Tensor):
        """Placeholder for quantum prediction.

        A quantum implementation can override this method or replace
        the class with the QML counterpart.
        """
        raise NotImplementedError(
            "Quantum prediction requires a quantum backend.  Import "
            "the QML module to use the quantum circuit."
        )


__all__ = ["QuantumClassifierModel"]
