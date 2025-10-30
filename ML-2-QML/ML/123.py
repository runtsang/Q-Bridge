"""Enhanced classical‑quantum hybrid model with feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QuantumNATEnhanced(nn.Module):
    """Classical CNN + fully‑connected head with early‑stopping support.

    The network processes 28×28 grayscale images, extracts features through
    three convolutional blocks, then projects to a four‑dimensional output
    suitable for multi‑task classification.  A simple early‑stopping
    mechanism is embedded to allow training scripts to terminate once the
    validation loss stops improving.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        dropout: float = 0.5,
        patience: int = 5,
    ) -> None:
        super().__init__()
        self.patience = patience
        self.best_val_loss = float("inf")
        self.counter = 0

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        self.batchnorm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        logits = self.fc(x)
        return self.batchnorm(logits)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return cross‑entropy loss for the four‑dimensional output."""
        return F.cross_entropy(logits, targets)

    def update_early_stopping(self, val_loss: float) -> bool:
        """Update early‑stopping status.

        Returns True if training should stop.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def lr_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Return a ReduceLROnPlateau scheduler that matches the early‑stopping."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.patience // 2, verbose=True
        )
