"""Hybrid classical model inspired by Quantum‑NAT, QCNN, and kernel methods.

This module defines :class:`HybridQuantumNAT`, a convolutional feature extractor followed by a
radial‑basis‑function kernel and a sigmoid head.  It can be used as a drop‑in replacement for
the classical counterpart of the hybrid quantum architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumNAT(nn.Module):
    """Classical convolutional network with an RBF kernel head."""
    def __init__(self, shift: float = 0.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 64)
        self.shift = shift
        self.gamma = gamma
        self.support_vectors: torch.Tensor | None = None

    def set_support_vectors(self, sv: torch.Tensor) -> None:
        """Store the support vectors for the RBF kernel."""
        self.support_vectors = sv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        if self.support_vectors is not None:
            # Compute pairwise squared Euclidean distance and apply RBF
            dists = torch.cdist(x, self.support_vectors, p=2)
            k = torch.exp(-self.gamma * dists.pow(2))
            out = k
        else:
            out = x
        probs = torch.sigmoid(out + self.shift)
        return probs

__all__ = ["HybridQuantumNAT"]
