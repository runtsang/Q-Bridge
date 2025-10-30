"""Hybrid classical kernel module combining CNN feature extraction with RBF kernel.

The implementation extends the original RBF kernel by first transforming raw
images through a lightweight convolutional network.  The extracted
4‑dimensional feature vectors are then fed into an exponential RBF
kernel.  The class is fully compatible with the original
``QuantumKernelMethod`` API, but adds feature learning and
batch‑normalisation, allowing it to be used seamlessly in downstream
kernel‑based learning pipelines.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Convolutional feature extractor producing a 4‑dimensional vector."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        out = self.conv(x)
        out = out.view(bsz, -1)
        out = self.fc(out)
        return self.bn(out)


class HybridQuantumKernel(nn.Module):
    """Classical RBF kernel with CNN‑based feature extraction."""
    def __init__(self, gamma: float = 1.0, feature_extractor: nn.Module | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two batches of images."""
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)
        diff = fx - fy
        dist2 = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * dist2)

    def kernel_matrix(self, a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two sequences of images."""
        mat = torch.zeros(len(a), len(b), device=a[0].device)
        for i, xa in enumerate(a):
            fx = self.feature_extractor(xa)
            for j, yb in enumerate(b):
                fy = self.feature_extractor(yb)
                mat[i, j] = torch.exp(-self.gamma * torch.sum((fx - fy) ** 2))
        return mat.cpu().numpy()


__all__ = ["FeatureExtractor", "HybridQuantumKernel"]
