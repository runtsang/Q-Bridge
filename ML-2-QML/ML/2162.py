"""QuantumHybridClassifier: classical variant with an MLP head.

This module implements a CNN backbone followed by a multi‑layer perceptron
head.  The MLP uses dropout and a sigmoid output to produce a binary
probability vector.

The architecture extends the seed by adding multiple hidden layers and
dropout to improve regularisation.  The final classification head is a
learnable MLP rather than a single linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Deep MLP with dropout used as the classification head."""

    def __init__(self, in_features: int, hidden_dims: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumHybridClassifier(nn.Module):
    """Classical CNN followed by an MLP head for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10‑dimensional feature vector

        # MLP head
        self.head = MLPHead(in_features=10, hidden_dims=[64, 32], dropout=0.2)

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

        logits = self.head(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumHybridClassifier"]
