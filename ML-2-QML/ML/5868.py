"""Purely classical hybrid model for binary classification.

This module implements the classical counterpart of the hybrid quantum
networks.  The architecture mirrors the quantum version but replaces the
parameterised quantum circuit with a learnable dense head that mimics
the quantum expectation layer.  The implementation is fully PyTorch
based and can be used as a drop‑in replacement for the quantum model
during ablation studies or as a baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalHybridHead(nn.Module):
    """Dense head that emulates the behaviour of a quantum expectation layer.

    The head consists of a linear transformation followed by a
    differentiable sigmoid that introduces a learnable bias term.
    """

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        # Shifted sigmoid to match the quantum output range
        return torch.sigmoid(logits + self.shift)


class HybridQCNet(nn.Module):
    """Classical CNN followed by a hybrid dense head for binary classification."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional backbone identical to the quantum version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical hybrid head
        self.head = ClassicalHybridHead(1, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1)."""
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1)


__all__ = ["HybridQCNet"]
