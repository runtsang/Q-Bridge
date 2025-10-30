"""Enhanced classical‑quantum hybrid model with dual‑branch processing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QuanvolutionDual(nn.Module):
    """Hybrid model with two parallel feature extraction paths: a classical
    convolution branch and a quantum‑kernel emulation branch.
    The two feature streams are concatenated before a final linear head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_out_channels: int = 4,
        conv_kernel_size: int = 2,
        conv_stride: int = 2,
        qkernel_dim: int = 4,
        qkernel_layers: int = 2,
    ) -> None:
        super().__init__()
        # Classical convolution branch
        self.conv_branch = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
        )
        # Quantum‑kernel emulation branch (a small MLP)
        self.qkernel_branch = nn.Sequential(
            nn.Flatten(),
            *([nn.Linear(28 * 28, 128), nn.ReLU()] * qkernel_layers),
            nn.Linear(128, qkernel_dim),
        )
        # Fusion head
        conv_feat_dim = conv_out_channels * 14 * 14
        self.fc = nn.Linear(conv_feat_dim + qkernel_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        conv_feat = self.conv_branch(x)
        conv_flat = conv_feat.view(x.size(0), -1)
        # Quantum‑kernel emulation branch
        qkernel_feat = self.qkernel_branch(x)
        # Concatenate and classify
        fused = torch.cat([conv_flat, qkernel_feat], dim=1)
        logits = self.fc(fused)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionDual"]
