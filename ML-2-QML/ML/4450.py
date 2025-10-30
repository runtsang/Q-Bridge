"""Hybrid binary classifier – classical implementation.

This module provides a CNN backbone followed by either a classical dense head or an
optional quantum‑kernel mapping.  The design mirrors the original `ClassicalQuantumBinaryClassification`
file but adds a configurable kernel layer and a clean API for switching between
classical and quantum inference paths.
"""

from __future__ import annotations

from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Classical hybrid head
# --------------------------------------------------------------------------- #
class HybridFunction(nn.Module):
    """Simple sigmoid head with an optional shift."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)


class Hybrid(nn.Module):
    """Dense layer followed by `HybridFunction`."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


# --------------------------------------------------------------------------- #
#  Classical CNN backbone
# --------------------------------------------------------------------------- #
class ClassicalCNN(nn.Module):
    """CNN feature extractor identical to the original seed."""
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
        return x  # raw logits


# --------------------------------------------------------------------------- #
#  Classical RBF kernel utilities
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """RBF kernel between two 1‑D tensors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that normalises inputs to 2‑D tensors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                 gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sequences of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
#  Hybrid binary classifier with optional quantum‑kernel feature
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    Main entry point.

    Parameters
    ----------
    use_kernel : bool
        If True, the output of the CNN is mapped through a quantum‑kernel
        layer before classification.
    kernel_gamma : float
        RBF kernel bandwidth used when `use_kernel` is True.
    """
    def __init__(self, use_kernel: bool = False, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.cnn = ClassicalCNN()
        self.hybrid_head = Hybrid(1)

        if use_kernel:
            self.kernel = Kernel(kernel_gamma)
            self.kernel_prototypes: Optional[torch.Tensor] = None
            self.kernel_classifier: Optional[nn.Linear] = None
        else:
            self.kernel = None

    def set_kernel_prototypes(self, prototypes: torch.Tensor) -> None:
        """Provide a set of prototype vectors for the kernel mapping."""
        if not self.use_kernel:
            raise RuntimeError("Kernel mapping is disabled.")
        self.kernel_prototypes = prototypes
        self.kernel_classifier = nn.Linear(prototypes.shape[0], 1).to(prototypes.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.cnn(x)          # shape (batch, 1)
        probs = torch.sigmoid(logits)  # shape (batch, 1)

        if self.use_kernel and self.kernel_prototypes is not None:
            # Compute kernel vector for each sample
            kvecs = torch.stack([self.kernel(probs, proto) for proto in self.kernel_prototypes], dim=-1)
            logits = self.kernel_classifier(kvecs)
            probs = torch.sigmoid(logits)

        return torch.cat((probs, 1 - probs), dim=-1)  # shape (batch, 2)


__all__ = ["HybridBinaryClassifier", "Kernel", "kernel_matrix"]
