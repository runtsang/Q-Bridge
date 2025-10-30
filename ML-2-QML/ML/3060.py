"""HybridQCNet – Classical implementation of the hybrid quantum‑binary classifier.

The architecture mirrors the original hybrid network but replaces the quantum
kernel by a classical RBF kernel and the quantum expectation head by a
differentiable sigmoid layer.  The module can be used as a drop‑in
replacement for the quantum version during development or when a quantum
backend is unavailable.

Key components:
* `RBFKernel` – classical radial‑basis function kernel, compatible with
  TorchQuantum's interface.
* `HybridLayer` – simple linear layer followed by a sigmoid that mimics the
  quantum expectation head.
* `HybridQCNet` – convolutional backbone + kernel layer + hybrid head.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFKernel(nn.Module):
    """Classical RBF kernel that emulates TorchQuantum's `Kernel` interface.

    Parameters
    ----------
    support_vectors : Iterable[torch.Tensor]
        Collection of vectors that form the kernel feature map.
    gamma : float
        Width parameter of the RBF kernel.
    """

    def __init__(self, support_vectors: Iterable[torch.Tensor], gamma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("support", torch.stack(list(support_vectors)))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return kernel vector for input `x`.

        The returned tensor has shape ``(len(support),)`` and contains the
        similarity of `x` to each support vector.
        """
        diff = self.support.unsqueeze(0) - x.unsqueeze(1)  # (1, n, d) - (1, 1, d)
        sq_norm = torch.sum(diff * diff, dim=-1)  # (1, n)
        return torch.exp(-self.gamma * sq_norm).squeeze(0)


class HybridLayer(nn.Module):
    """Differentiable head that replaces the quantum expectation circuit.

    It consists of a linear projection followed by a sigmoid activation,
    optionally with a learnable shift.
    """

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) + self.shift
        return torch.sigmoid(logits)


class HybridQCNet(nn.Module):
    """Full hybrid neural network with a classical kernel and hybrid head.

    The network is compatible with the quantum version in terms of
    input/output signatures and can be used interchangeably.
    """

    def __init__(self, support_vectors: Iterable[torch.Tensor], gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical kernel layer
        self.kernel = RBFKernel(support_vectors, gamma=gamma)

        # Hybrid head
        self.hybrid = HybridLayer(in_features=self.kernel.support.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
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

        # Kernel feature map
        kernel_features = self.kernel(x)

        # Hybrid head
        probs = self.hybrid(kernel_features)

        # Return two‑class probability vector
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["RBFKernel", "HybridLayer", "HybridQCNet"]
