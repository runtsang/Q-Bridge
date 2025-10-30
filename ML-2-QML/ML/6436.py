"""Hybrid classical model combining CNN feature extraction and a quantum-inspired projection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNATModel(nn.Module):
    """Classical CNN with a quantum‑inspired linear projection head.

    The architecture merges ideas from the original QFCModel (convolutional feature
    extractor + fully connected head) and the QuanvolutionFilter (patch‑based
    processing).  Instead of a true quantum kernel, the head performs a
    randomized linear projection that mimics the statistics of a quantum
    measurement.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Convolutional backbone: two conv layers similar to QFCModel
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Patch extraction: split the feature map into 2×2 patches
        self.patch_size = 2
        # Randomized linear projection to emulate quantum measurement
        self.proj = nn.Linear(16 * self.patch_size * self.patch_size, 64, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        self.relu = nn.ReLU(inplace=True)
        # Final classification head
        self.classifier = nn.Linear(64, num_classes)
        self.norm = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feat = self.features(x)  # [bsz, 16, H', W']
        # Reshape into patches
        _, c, h, w = feat.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        feat = feat.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # feat shape: [bsz, 16, H'/ps, W'/ps, ps, ps]
        feat = feat.contiguous().view(bsz, -1, self.patch_size * self.patch_size * c)
        # Random projection per patch
        proj_feat = self.proj(feat)  # [bsz, num_patches, 64]
        proj_feat = proj_feat.mean(dim=1)  # aggregate patches
        proj_feat = self.relu(proj_feat)
        proj_feat = self.norm(proj_feat)
        logits = self.classifier(proj_feat)
        return logits


__all__ = ["HybridNATModel"]
