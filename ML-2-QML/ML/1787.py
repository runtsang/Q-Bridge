"""Classical counterpart of the hybrid quantum binary classifier.

Provides a deep CNN backbone followed by a learnable dense head that
mimics the quantum expectation. The head uses a GELU activation and
a learnable bias shift, allowing the model to approximate the sigmoid
behaviour without a quantum backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalHybridHead(nn.Module):
    """Dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(shift))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

class HybridQCNet(nn.Module):
    """CNN-based binary classifier mirroring the structure of the quantum model."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = ClassicalHybridHead(self.fc3.out_features, shift=0.0)
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

__all__ = ["HybridQCNet"]
