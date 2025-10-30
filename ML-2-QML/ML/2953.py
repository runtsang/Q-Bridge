"""Classical hybrid binary classifier with CNN backbone and fully connected head.

This module implements a robust classical network that mirrors the structure of the quantum
hybrid model from the anchor reference, but replaces the quantum expectation head with a
simple sigmoid activation.  The design incorporates a projection network inspired by the
EstimatorQNN example, providing a lightweight feature reduction before the final
classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """
    Classical CNN followed by a tiny fully‑connected projection and sigmoid head.
    The architecture is a direct analogue of the quantum hybrid model, enabling
    side‑by‑side experimentation while staying fully differentiable in PyTorch.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone (same as the quantum version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Projection to a single scalar (EstimatorQNN style)
        self.proj = nn.Linear(1, 1, bias=False)  # identity mapping, kept for symmetry
        # Sigmoid head mimicking the quantum expectation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Fully‑connected reduction
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # shape (batch, 1)

        # Projection (identity) then sigmoid
        x = self.proj(x)         # shape (batch, 1)
        probs = self.sigmoid(x)
        # Return two‑class probability vector
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
