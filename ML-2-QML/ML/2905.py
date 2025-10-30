"""Hybrid classical-quantum quanvolution model.

This module defines a `QuanvolutionEstimator` that combines a 2×2
convolutional filter with a lightweight quantum patch encoder inspired
by the EstimatorQNN example.  The quantum part is emulated by a small
neural module so that the whole network remains pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class _QuantumPatchEmulator(nn.Module):
    """Lightweight emulation of a quantum patch encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch shape: (batch, 4)
        return self.linear(patch)


class QuanvolutionEstimator(nn.Module):
    """Hybrid quanvolution model."""
    def __init__(self) -> None:
        super().__init__()
        # Classical branch – identical to the original seed
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Quantum branch – emulated
        self.quantum_patch = _QuantumPatchEmulator()
        # Linear head: 4 * 14 * 14 classical + 4 * 14 * 14 quantum = 4 * 14 * 14 * 2
        self.linear = nn.Linear(4 * 14 * 14 * 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        # Classical features
        cls_feat = self.conv(x).view(x.size(0), -1)

        # Quantum features – iterate over 2×2 patches
        bsz = x.shape[0]
        patches = []
        img = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[:, r, c:r+2, c+2:c+4].reshape(bsz, 4)
                qfeat = self.quantum_patch(patch)
                patches.append(qfeat)
        qfeat = torch.cat(patches, dim=1)

        # Concatenate and classify
        combined = torch.cat([cls_feat, qfeat], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)
