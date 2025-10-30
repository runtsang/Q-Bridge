"""
DualHybridNet – classical dense head with optional dropout and a learnable weighting
between classical and quantum contributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHybridNet(nn.Module):
    """Classical CNN with a lightweight MLP head.

    The architecture mirrors the original hybrid model but replaces the
    quantum expectation layer with a fully‑differentiable dense head.
    This provides a fast baseline that can be used for ablation studies
    or as a drop‑in replacement when a quantum backend is not available.
    """

    def __init__(self, dropout: float = 0.0, hidden_dim: int = 64) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical head
        self.classical_head = nn.Sequential(
            nn.Linear(self.fc3.out_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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
        logits = self.classical_head(x)
        prob = torch.sigmoid(logits)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["DualHybridNet"]
