"""Enhanced classical sampler neural network with training utilities.

This module implements a deeper feed‑forward network with batch
normalisation, dropout and a convenience training method.  It
retains the original interface: a top‑level function `create_sampler_qnn`
that returns an instantiated `SamplerQNN` class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SamplerQNN(nn.Module):
    """A robust two‑class sampler network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing class probabilities."""
        return F.softmax(self.net(inputs), dim=-1)

    def train_on_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """Convenience training routine for binary classification."""
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            logits = self.net(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:03d}/{epochs:03d} – loss: {loss.item():.6f}")

def create_sampler_qnn() -> SamplerQNN:
    """Return a freshly instantiated sampler network."""
    return SamplerQNN()

__all__ = ["SamplerQNN"]
