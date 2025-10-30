"""QuantumHybridClassifier – classical extension of the original binary classifier.

This module mirrors the original architecture but replaces the simple
activation head with a trainable fully‑connected layer that
produces a score vector.  The output is then fed into a
`Softmax` layer so that the model can be trained end‑to‑end with
cross‑entropy loss.  The design also introduces a
`FeatureExtractor` that re‑uses the convolutional backbone and
provides a shared representation for downstream heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeatureExtractor", "LinearHead", "QuantumHybridClassifier"]


class FeatureExtractor(nn.Module):
    """Extracts features from input images using the convolutional backbone."""
    def __init__(self, in_channels: int = 3, output_dim: int = 120) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # The flattened dimension depends on input image size; we keep it flexible
        self.fc1 = nn.Linear(55815, output_dim)
        self.fc2 = nn.Linear(output_dim, 84)
        self.fc3 = nn.Linear(84, 1)  # Output a single scalar per sample

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
        return x


class LinearHead(nn.Module):
    """A simple two‑class classifier head."""
    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuantumHybridClassifier(nn.Module):
    """A purely classical binary classifier with a softmax head."""
    def __init__(self) -> None:
        super().__init__()
        self.extractor = FeatureExtractor()
        self.head = LinearHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        logits = self.head(features)
        probs = F.softmax(logits, dim=-1)
        return probs
