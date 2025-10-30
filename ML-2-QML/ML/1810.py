"""Enhanced classical estimator with modern regularisation and training utilities."""
from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ["EstimatorQNN"]


class EstimatorQNN(nn.Module):
    """
    A small but flexible fully‑connected regressor.

    Architecture:
        - Input → Linear(2→64) → BatchNorm → ReLU
        - Dropout(0.3)
        - Linear(64→32) → BatchNorm → ReLU
        - Linear(32→1)

    The network is intentionally over‑parameterised for the toy task
    but includes dropout and batch‑norm to demonstrate regularisation.
    """

    def __init__(self, input_dim: int = 2, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Utility helpers (not required for the seed but useful for experiments)
    # ------------------------------------------------------------------
    def predict(self, x: Tensor, device: str | None = None) -> Tensor:
        """Convenience wrapper that moves data to the correct device."""
        self.eval()
        if device:
            x = x.to(device)
        with torch.no_grad():
            return self(x)

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Mean‑squared error loss."""
        return F.mse_loss(pred, target)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        data: Tensor,
        target: Tensor,
        device: str | None = None,
    ) -> Tensor:
        """Single optimizer step returning the loss value."""
        self.train()
        if device:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = self(data)
        loss = self.loss(pred, target)
        loss.backward()
        optimizer.step()
        return loss


def EstimatorQNNFactory() -> EstimatorQNN:
    """Factory function mirroring the original anchor API."""
    return EstimatorQNN()
