"""Classical deep‑learning counterpart to the enhanced hybrid classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VariationalDenseHead(nn.Module):
    """Dense head that mimics a variational quantum layer."""
    def __init__(self, in_features: int, out_features: int = 2, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))

class QuantumHybridClassifier(nn.Module):
    """Classical CNN followed by a variational dense head."""
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.head = VariationalDenseHead(num_classes, out_features=num_classes)

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
        return self.head(x)

__all__ = ["VariationalDenseHead", "QuantumHybridClassifier"]
