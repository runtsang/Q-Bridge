"""QuantumNATEnhanced: Classical backbone with multi‑scale feature extraction and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class QuantumNATEnhanced(nn.Module):
    """Hybrid‑style classical network that leverages multi‑scale feature maps and residual connections."""
    def __init__(self, in_channels: int = 1, out_features: int = 4, hidden_dim: int = 128):
        super().__init__()
        # Feature extractor: Conv + residual blocks
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
            nn.MaxPool2d(2),
        )
        # Multi‑scale pooling
        self.avg_pool_small = nn.AdaptiveAvgPool2d((4, 4))
        self.avg_pool_large = nn.AdaptiveAvgPool2d((2, 2))
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4 + 16 * 2 * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, out_features)
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        # Multi‑scale pooling
        pooled_small = self.avg_pool_small(features).view(bsz, -1)
        pooled_large = self.avg_pool_large(features).view(bsz, -1)
        pooled = torch.cat([pooled_small, pooled_large], dim=1)
        out = self.fc(pooled)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
