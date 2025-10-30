"""QuantumNATModel: Classical attention‑augmented CNN with residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    """Self‑attention over flattened spatial features."""
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        seq_len = height * width
        # Flatten spatial dims
        x_flat = x.view(batch, channels, seq_len).transpose(1, 2)  # (batch, seq_len, channels)
        x_norm = self.norm(x_flat)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        # Reshape back
        attn_output = attn_output.transpose(1, 2).view(batch, channels, height, width)
        return attn_output

class QuantumNATModel(nn.Module):
    """Attention‑augmented CNN with residual connections and a final linear projection."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(8, 16, stride=2)
        self.attn = SelfAttention(embed_dim=16)
        # Classifier
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.attn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.norm(x)

__all__ = ["QuantumNATModel"]
