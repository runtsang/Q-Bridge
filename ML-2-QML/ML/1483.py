"""Enhanced classical model with residual blocks and optional quantum encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv‑bn‑ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential() if in_ch == out_ch and stride == 1 else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class QuantumNATEnhanced(nn.Module):
    """
    Classical CNN with optional quantum encoder.

    Parameters
    ----------
    use_quantum : bool
        If True, the 6‑pixel pooled feature vector is fed into a variational
        quantum circuit (see :class:`QuantumEncoder`) instead of a linear
        projection.  The quantum encoder returns a 4‑dimensional vector
        matching the original model's output shape.
    """
    def __init__(self, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum

        # Feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8, 16, stride=2),
            ResidualBlock(16, 32, stride=2),
        )

        if self.use_quantum:
            from.quantum_encoder import QuantumEncoder
            self.encoder = QuantumEncoder()
        else:
            self.encoder = nn.Sequential(
                nn.Linear(32 * 7 * 7, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 4)
            )

        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                     # shape: (B,32,7,7)
        if self.use_quantum:
            # Flatten and feed into quantum circuit
            pooled = F.avg_pool2d(x, 6).view(x.size(0), -1)  # (B,32)
            out = self.encoder(pooled)                       # (B,4)
        else:
            flat = x.view(x.size(0), -1)                     # (B,32*7*7)
            out = self.encoder(flat)                        # (B,4)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
