"""Enhanced classical binary classifier with a hybrid-style head.

The architecture mirrors the original hybrid model but replaces the quantum
block with a purely classical parametric head.  Dropout and batch
normalisation provide improved regularisation, and the head uses a
learnable shift before sigmoid to mimic a quantum expectation value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricLogit(nn.Module):
    """Linear layer followed by a learnable shift before sigmoid.

    Mimics the behaviour of a quantum expectation value but remains fully
    differentiable within PyTorch.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) + self.shift
        return torch.sigmoid(logits)


class HybridBinaryClassifier(nn.Module):
    """ConvNet + fully connected layers + ParametricLogit head.

    The structure mirrors the original hybrid architecture but replaces the
    quantum block with a classical parametric head that retains the same
    interface.  Dropout and batch normalisation provide improved regularisation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout2d(p=0.3)

        self.fc1 = nn.Linear(55815, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 1)

        self.head = ParametricLogit(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)

        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ParametricLogit", "HybridBinaryClassifier"]
