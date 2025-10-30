"""Classical PyTorch implementation of a hybrid quantum‑classical binary classifier.

This module mirrors the architecture of the original QCNet but replaces the quantum head with a
differentiable sigmoid layer.  It also offers a drop‑in classical convolutional filter that can
stand in for the quantum quanvolution used in the QML version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical sigmoid head that emulates the quantum expectation
class HybridFunction(nn.Module):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(inputs + self.shift)

# Classical convolutional filter (drop‑in replacement for quanvolution)
class ConvFilter(nn.Module):
    """Simple 2‑D convolution that mimics the behaviour of the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)

class HybridQuantumClassifier(nn.Module):
    """Classical CNN followed by a sigmoid head, optionally using a classical filter."""
    def __init__(self,
                 use_filter: bool = False,
                 filter_kernel: int = 2,
                 filter_threshold: float = 0.0,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.use_filter = use_filter
        self.filter = ConvFilter(filter_kernel, filter_threshold) if use_filter else None

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridFunction(shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        if self.use_filter:
            # Convert to single‑channel for the filter
            x = torch.mean(x, dim=1, keepdim=True)
            x = self.filter(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.head(x)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["HybridQuantumClassifier", "HybridFunction", "ConvFilter"]
