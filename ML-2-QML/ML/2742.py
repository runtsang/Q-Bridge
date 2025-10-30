"""Hybrid classical neural network combining a patch‑wise quantum‑kernel approximation with a conventional CNN.

The network first maps each 2×2 image patch through a classical random projection that mimics a 4‑qubit quantum kernel.  The resulting 4‑channel feature map is processed by two convolutional layers followed by a fully‑connected head, producing a 4‑dimensional output that is batch‑normalised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQuantumKernel(nn.Module):
    """Approximate a 4‑qubit random layer with a classical random projection.

    The kernel maps a 4‑dimensional patch to a 4‑dimensional feature vector
    using a random orthogonal matrix followed by a non‑linearity.
    """
    def __init__(self, in_dim: int = 4, out_dim: int = 4, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((in_dim, out_dim))
        q, _ = np.linalg.qr(mat)
        self.weight = nn.Parameter(torch.tensor(q, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        return torch.sin(out) + torch.cos(out)

class PatchQuantumFeatureExtractor(nn.Module):
    """Apply the ClassicalQuantumKernel to every 2×2 patch of a grayscale image."""
    def __init__(self):
        super().__init__()
        self.kernel = ClassicalQuantumKernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, H, W = x.shape
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        patches = patches.contiguous().view(bsz, H // 2, W // 2, -1)
        patches = patches.view(-1, 4)
        feats = self.kernel(patches)
        feats = feats.view(bsz, H // 2, W // 2, 4)
        return feats.permute(0, 3, 1, 2)

class QFCModel(nn.Module):
    """Hybrid classical network combining a patch‑wise quantum kernel,
    a conventional CNN, and a fully‑connected head.

    Architecture:
        - PatchQuantumFeatureExtractor: 2×2 patches → 4‑channel feature map
        - Conv block 1: 4→8 channels, 3×3 kernel
        - Conv block 2: 8→16 channels, 3×3 kernel
        - Global average pooling
        - FC head: 16 → 64 → 4
        - BatchNorm on output
    """
    def __init__(self) -> None:
        super().__init__()
        self.patch_qf = PatchQuantumFeatureExtractor()
        self.features = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * (28 // 4) * (28 // 4), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.patch_qf(x)  # (bsz, 4, 14, 14)
        x = self.features(x)  # (bsz, 16, 7, 7)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["QFCModel"]
