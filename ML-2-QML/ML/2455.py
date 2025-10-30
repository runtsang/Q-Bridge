"""Hybrid classical-quantum inspired binary classification/regression network.

This module defines a purely classical PyTorch implementation that mirrors the
architectural design of the quantum‑augmented network in the anchor reference.
It supports both binary classification and regression tasks and replaces the
quantum expectation head with a differentiable sigmoid or linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQCNet(nn.Module):
    """Purely classical network that mimics the hybrid QCNet architecture.

    Parameters
    ----------
    mode : {"classification", "regression"}
        Determines the type of head used.  For classification a sigmoid
        activation is applied to produce a probability; for regression the
        output is left linear.
    """

    def __init__(self, mode: str = "classification") -> None:
        super().__init__()
        self.mode = mode
        # Convolutional feature extractor (identical to the quantum‑augmented version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Classical head that emulates the quantum expectation layer
        if self.mode == "classification":
            self.head = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid(),
            )
        else:
            self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.head(x)
        if self.mode == "classification":
            # Convert single probability into a two‑class distribution
            return torch.cat((x, 1 - x), dim=-1)
        return x.squeeze(-1)


__all__ = ["HybridQCNet"]
