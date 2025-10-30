"""Hybrid classical model combining CNN feature extraction with a quantum-inspired kernel layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridQuantumNAT(nn.Module):
    """
    Classical CNN + RBF kernel hybrid model.
    Features are extracted via a shallow CNN, flattened, and compared
    against a set of trainable prototype vectors using an RBF kernel.
    The kernel similarities are then linearly mapped to the final
    four-dimensional output.  This architecture mirrors the quantum
    variant but remains fully classical, enabling fast inference
    on CPUs/GPUs.
    """

    def __init__(self, n_prototypes: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Reduce to a fixed dimensionality for kernel comparison
        self.fc_reduce = nn.Linear(16 * 7 * 7, 64)
        # Trainable prototypes in the reduced space
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, 64))
        # Kernel hyper‑parameter
        self.gamma = nn.Parameter(torch.tensor(gamma))
        # Final linear map from kernel similarities to output
        self.fc_out = nn.Linear(n_prototypes, 4)
        self.norm = nn.BatchNorm1d(4)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim), y: (n_prototypes, dim)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (batch, n, dim)
        dist_sq = (diff * diff).sum(dim=-1)  # (batch, n)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        reduced = F.relu(self.fc_reduce(flattened))
        # Compute kernel similarities to prototypes
        kernel_sim = self._rbf_kernel(reduced, self.prototypes)
        out = self.fc_out(kernel_sim)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
