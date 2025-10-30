"""Hybrid classical binary classifier with quantum-inspired layers.

This module implements a CNN backbone followed by a classical approximation
of a quantum kernel feature map and a simple linear head.  The kernel
is realised with random Fourier features, providing a scalable
classical surrogate for the quantum kernel used in the QML variant.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ClassicalKernelFeatureMap(nn.Module):
    """Random Fourier feature map approximating a Gaussian quantum kernel."""
    def __init__(self, input_dim: int, output_dim: int = 64, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * np.sqrt(2 * gamma))
        self.bias = nn.Parameter(torch.randn(output_dim) * 2 * np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.weights) + self.bias
        return torch.cos(z)


class HybridHead(nn.Module):
    """Linear head producing class probabilities."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


class HybridBinaryClassifier(nn.Module):
    """Hybrid classifier combining a CNN backbone, a quantum‑kernel surrogate
    and a linear head."""
    def __init__(
        self,
        use_kernel: bool = True,
        kernel_dim: int = 64,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        dummy = torch.zeros(1, 3, 32, 32)
        flat_dim = self.backbone(dummy).view(1, -1).size(1)
        self.flatten = nn.Flatten()
        self.kernel = ClassicalKernelFeatureMap(flat_dim, kernel_dim, gamma) if use_kernel else None
        head_input = kernel_dim if use_kernel else flat_dim
        self.head = HybridHead(head_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        if self.kernel is not None:
            x = self.kernel(x)
        return self.head(x)


__all__ = ["HybridBinaryClassifier"]
