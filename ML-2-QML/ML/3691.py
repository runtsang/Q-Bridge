"""Fully classical QuanvolutionSamplerNet with MLP-based sampler.

The network mirrors the quantum structure: a 2×2 patch extractor,
a lightweight MLP that mimics the quantum sampler, and a linear
classification head.  This design allows direct comparison with the
quantum variant and serves as a robust baseline for hybrid studies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionSamplerNet(nn.Module):
    """Classical approximation of a quanvolution filter with a sampler‑like MLP.

    The architecture follows the same patch‑wise pipeline as the quantum
    implementation:
        1. Convolutional patch extraction (2×2 stride‑2)
        2. Patch‑wise MLP that produces 4‑dimensional feature vectors
           (analogous to a quantum measurement)
        3. Linear classifier
    """

    def __init__(self) -> None:
        super().__init__()
        # 2×2 patches → 4 channels
        self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # MLP that emulates the quantum sampler
        self.patch_mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 2×2 patches via convolution
        patch_feats = self.patch_conv(x)  # shape (B, 4, 14, 14)
        # Reshape to (B, 14*14, 4) for per‑patch processing
        batch, ch, h, w = patch_feats.shape
        patches = patch_feats.permute(0, 2, 3, 1).reshape(batch, h * w, ch)
        # Apply the sampler‑MLP to each patch
        patch_out = self.patch_mlp(patches)  # (B, 14*14, 4)
        # Flatten for classification
        flat = patch_out.reshape(batch, -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionSamplerNet"]
