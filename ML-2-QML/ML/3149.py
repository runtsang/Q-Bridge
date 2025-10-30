"""
Hybrid classical model combining quanvolutional filtering with a
fully‑connected projection, inspired by Quantum‑NAT and the quanvolution
example.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classic 2×2 patch filter that mimics the behaviour of the quantum
    quanvolution.  Each patch is flattened and passed through a 2×2
    convolution, then reshaped into a vector of length 4.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Apply the 2×2 convolution and flatten to (batch, 4*14*14)
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridQuantumNAT(nn.Module):
    """
    Classical hybrid architecture:
        - QuanvolutionFilter extracts local features.
        - A 2‑layer fully‑connected head projects to 4 outputs,
          followed by batch‑norm.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # 4 * 14 * 14 = 784 input features
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        out = self.fc(features)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
