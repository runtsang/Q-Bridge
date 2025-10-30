"""Hybrid kernel‑based binary classifier using a classical neural network backbone.

The model consists of a CNN feature extractor, an RBF kernel embedding against a
learnable support set, and a linear sigmoid head.  The architecture mirrors the
original ``ClassicalQuantumBinaryClassification`` design but replaces the quantum
circuit with a classical kernel module, allowing pure PyTorch training while
retaining the kernel‑based expressivity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical RBF kernel implementation from the shared module
from QuantumKernelMethod import Kernel


class HybridKernelHead(nn.Module):
    """Linear head that maps kernel embeddings to a probability."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        probs = torch.sigmoid(logits + self.shift)
        return probs


class HybridKernelQCNet(nn.Module):
    """CNN → RBF kernel embedding → hybrid linear head."""

    def __init__(
        self,
        n_support: int = 16,
        gamma: float = 1.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Kernel embedding
        self.kernel = Kernel(gamma)
        # Learnable support vectors initialized randomly
        self.register_buffer("support", torch.randn(n_support, 1))

        # Hybrid head
        self.hybrid_head = HybridKernelHead(n_support, shift)

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
        x = self.fc3(x)

        # Compute kernel embedding against support vectors
        k = torch.stack([self.kernel(x, sv) for sv in self.support], dim=1)
        probs = self.hybrid_head(k)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridKernelQCNet"]
