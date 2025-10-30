"""QuantumHybridNet: Classical backbone + hybrid head.

This module merges ideas from the original Quantum‑NAT, the
Classical‑Quantum binary classifier, and the Deep‑ResNet
architecture.  It is fully classical, using only PyTorch
and NumPy.  The network can be trained with standard optimizers
and includes a flexible hybrid head that can be swapped with
a quantum circuit or a lightweight dense block.

Author: gpt-oss-20b
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["QuantumHybridNet", "ClassicHead", "HybridHead"]

# --------------------------------------------------------------------------- #
# Helper: depth‑wise separable convolution + residual block
# --------------------------------------------------------------------------- #
class _DepthSepConv(nn.Module):
    """Depth‑wise separable convolution with optional residual."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel,
                            stride=stride, padding=padding, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return self.relu(self.bn(out))

class _ResidualBlock(nn.Module):
    """Single residual block with depth‑wise separable conv."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _DepthSepConv(channels, channels)
        self.conv2 = _DepthSepConv(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# --------------------------------------------------------------------------- #
# Classical head used when no quantum circuit is provided.
# --------------------------------------------------------------------------- #
class ClassicHead(nn.Module):
    """Standard dense head for binary classification."""
    def __init__(self, in_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        return self.fc(x)

# --------------------------------------------------------------------------- #
# Hybrid head placeholder: can be swapped with a quantum module.
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """
    A lightweight head that forwards through a provided quantum module
    if supplied, otherwise falls back to ClassicHead.
    """
    def __init__(self, in_features: int, quantum_head: nn.Module | None = None, num_classes: int = 2) -> None:
        super().__init__()
        self.quantum_head = quantum_head
        self.classic_head = ClassicHead(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantum_head is not None:
            return self.quantum_head(x)
        return self.classic_head(x)

# --------------------------------------------------------------------------- #
# Main hybrid network
# --------------------------------------------------------------------------- #
class QuantumHybridNet(nn.Module):
    """
    Convolutional backbone followed by a flexible hybrid head.

    Parameters
    ----------
    use_quantum_head : bool
        Whether to use a quantum head.  The quantum module must be
        passed via ``quantum_head``.
    quantum_head : nn.Module | None
        Instance of a quantum module that accepts a 1‑D tensor
        of shape (batch, in_features) and returns logits.
    """
    def __init__(self,
                 use_quantum_head: bool = False,
                 quantum_head: nn.Module | None = None,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            _ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        in_features = 128  # After adaptive pooling
        if use_quantum_head:
            if quantum_head is None:
                raise ValueError("quantum_head must be provided when use_quantum_head is True")
            self.head = HybridHead(in_features, quantum_head, num_classes)
        else:
            self.head = ClassicHead(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 3, H, W).

        Returns
        -------
        logits : torch.Tensor
            Logits of shape (batch, num_classes).
        """
        features = self.backbone(x)
        flat = self.flatten(features)
        logits = self.head(flat)
        return logits
