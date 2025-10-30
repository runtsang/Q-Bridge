"""Enhanced classical sampler network with training utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A two‑layer neural network that acts as a classical sampler.

    Layers:
        - Linear(2 → 4) with batch norm and ReLU
        - Dropout(0.3)
        - Linear(4 → 2)
    Returns a probability vector via softmax.

    The network comes with a small ``train_one_step`` helper that
    performs a single gradient update given a loss function.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4, 2, bias=False),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Custom weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution.

        Args:
            inputs: Tensor of shape (..., 2)

        Returns:
            Tensor of shape (..., 2) with probabilities that sum to 1.
        """
        return F.softmax(self.net(inputs), dim=-1)

    def train_one_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """
        Performs a single training step and returns the loss value.

        Args:
            inputs: Input tensor.
            targets: Target class indices.
            loss_fn: Loss function (default: CrossEntropyLoss).
            optimizer: Optimizer; if None, a default Adam with lr=1e-3 is created.

        Returns:
            Scalar loss value.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        optimizer.zero_grad()
        logits = self.net(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["SamplerQNN"]
