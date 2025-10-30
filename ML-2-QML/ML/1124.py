from __future__ import annotations

import torch
from torch import nn
from typing import Sequence

class EstimatorQNN(nn.Module):
    """
    A flexible feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : Sequence[int], default (8, 4)
        Sizes of hidden layers.
    dropout : float, default 0.0
        Dropout probability after each hidden layer.
    activation : nn.Module, default nn.Tanh()
        Activation function.
    output_scale : float, default 1.0
        Scale factor applied to the raw output.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (8, 4),
        dropout: float = 0.0,
        activation: nn.Module = nn.Tanh(),
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.output_scale = output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_scale * self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the network output."""
        with torch.no_grad():
            return self.forward(x)

    @staticmethod
    def from_config(cfg: dict) -> "EstimatorQNN":
        """Instantiate from a configuration dictionary."""
        return EstimatorQNN(
            input_dim=cfg.get("input_dim", 2),
            hidden_dims=tuple(cfg.get("hidden_dims", (8, 4))),
            dropout=cfg.get("dropout", 0.0),
            activation=cfg.get("activation", nn.Tanh()),
            output_scale=cfg.get("output_scale", 1.0),
        )

    def train_on_loader(
        self,
        loader,
        optimizer,
        loss_fn,
        epochs: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:
        """Convenience training loop for a PyTorch DataLoader."""
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = self(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss:.4f}")

__all__ = ["EstimatorQNN"]
