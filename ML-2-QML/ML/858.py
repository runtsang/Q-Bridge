"""Hybrid classical classifier with a variational post‑processing layer.

This module defines a single ``QuantumClassifierModel`` class that implements
- a deep feed‑forward network (depth 4, configurable) with *all* the
  weights and parameters that can be used.
- a variational post‑processing MLP that refines the logits before
  producing the final class probabilities.
The class exposes a ``train`` method that runs a standard optimizer loop
and a ``predict`` method that returns class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

class QuantumClassifierModel(nn.Module):
    """Hybrid classical classifier with a variational post‑processing layer.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int, default 4
        Number of hidden layers in the backbone.
    hidden_dim : int, default 128
        Width of each hidden layer.
    postprocess_layers : int, default 2
        Number of layers in the variational post‑processing MLP.
    regularization : float, default 1e-4
        L2 weight‑decay applied to all parameters.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 4,
        hidden_dim: int = 128,
        postprocess_layers: int = 2,
        regularization: float = 1e-4,
    ) -> None:
        super().__init__()
        backbone_layers = []
        in_dim = num_features
        for _ in range(depth):
            backbone_layers.append(nn.Linear(in_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
            in_dim = hidden_dim
        backbone_layers.append(nn.Linear(in_dim, 2))
        self.backbone = nn.Sequential(*backbone_layers)

        # Variational post‑processing MLP
        post_layers = []
        in_dim = 2
        for _ in range(postprocess_layers - 1):
            post_layers.append(nn.Linear(in_dim, hidden_dim))
            post_layers.append(nn.ReLU())
            in_dim = hidden_dim
        post_layers.append(nn.Linear(in_dim, 2))
        self.postprocess = nn.Sequential(*post_layers)

        self.regularization = regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits after backbone and post‑processing."""
        logits = self.backbone(x)
        return self.postprocess(logits)

    def train_model(
        self,
        data_loader,
        epochs: int = 20,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        """End‑to‑end training loop using Adam and cross‑entropy loss."""
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.regularization)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(data_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")

    def predict(self, x: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return class logits for input ``x``."""
        self.eval()
        with torch.no_grad():
            return self.forward(x.to(device))

__all__ = ["QuantumClassifierModel"]
