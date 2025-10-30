"""HybridQCNet: Classical CNN binary classifier with synthetic data generator.

The class provides a standard convolutional backbone followed by a learnable dense
head.  The synthetic data generator is inspired by the quantum regression
example and can be used for quick sanity checks or for pre‑training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQCNet(nn.Module):
    """Classical CNN + dense head for binary classification."""

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.classifier = nn.Sigmoid()

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
        probs = self.classifier(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

    @staticmethod
    def generate_synthetic_data(num_samples: int = 1000, num_features: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Generate a toy dataset that mimics the quantum regression example."""
        x = np.random.uniform(-1.0, 1.0, size=(num_samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        binary = (y > 0).astype(np.float32)
        return x, binary

__all__ = ["HybridQCNet"]
