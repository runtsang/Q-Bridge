"""Classical binary classifier mirroring the hybrid quantum network.

The network uses a lightweight feed‑forward head inspired by EstimatorQNN,
providing a differentiable sigmoid output that can be trained with standard
gradient descent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid(nn.Module):
    """Classical head that mimics the quantum expectation layer.

    It uses a small feed‑forward network followed by a sigmoid activation,
    echoing the structure of EstimatorQNN (Linear → tanh → Linear → tanh → Linear).
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return torch.sigmoid(logits + self.shift)

class QCNet(nn.Module):
    """CNN followed by the classical hybrid head."""
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
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        probs = self.hybrid(x)
        return torch.stack((probs, 1 - probs), dim=-1)

__all__ = ["Hybrid", "QCNet"]
