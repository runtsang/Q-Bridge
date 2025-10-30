"""Purely classical convolutional baseline for binary classification.

This module implements a CNN that mirrors the structure of the original
hybrid model but without any quantum layers. It can be used as a
performance baseline or when a quantum backend is unavailable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQCNNClassifier(nn.Module):
    """Classical CNN baseline for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.drop2 = nn.Dropout2d(p=0.55)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

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
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNNClassifier"]
