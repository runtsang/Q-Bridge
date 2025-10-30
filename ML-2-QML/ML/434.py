"""Enhanced classical model for Quantum‑NAT.

This model extends the original CNN+FC architecture with residual
blocks, batch‑normalisation and dropout, providing a richer feature
representation while keeping the same 4‑class output interface.

The class can be dropped into any PyTorch training loop without
modification of the forward signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two Conv‑BN‑ReLU layers and an optional
    down‑sampling path to match dimensions."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class QuantumNATEnhanced(nn.Module):
    """Classical hybrid‑style model that preserves the 4‑class output
    interface of the original seed while adding depth and regularisation."""
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: Conv → BN → ReLU → Pool, Residual → Pool
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16),
            nn.MaxPool2d(2),
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        logits = self.classifier(flat)
        return self.norm(logits)


__all__ = ["QuantumNATEnhanced"]
