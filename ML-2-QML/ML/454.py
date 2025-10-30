"""Classical-only implementation of a hybrid quantum classifier.

This module provides a pure PyTorch neural network that mirrors the
convolutional backbone and dense head of the original hybrid model.
The final layer uses a sigmoid activation as a classical proxy for
the quantum expectation value, allowing identical input shapes and
training pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQCNet(nn.Module):
    """Pure‑classical binary classifier that mirrors the structure of the
    hybrid quantum network.  The final dense layer is replaced with a
    sigmoid activation that serves as a stand‑in for the quantum
    expectation value."""
    def __init__(self,
                 in_features: int = 55815,
                 hidden_sizes: tuple[int, int] = (120, 84),
                 dropout: tuple[float, float] = (0.2, 0.5)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=dropout[0])
        self.drop2 = nn.Dropout2d(p=dropout[1])
        self.fc1 = nn.Linear(in_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self._sigmoid = nn.Sigmoid()

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
        prob = self._sigmoid(x)
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["HybridQCNet"]
