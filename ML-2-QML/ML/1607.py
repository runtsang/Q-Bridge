"""
Extended classical CNN with residual connections, attention, and dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A lightweight residual block for 2‑D feature maps."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out += self.shortcut(x)
        return F.relu(out)


class QFCModel(nn.Module):
    """CNN ➜ Attention ➜ FC pipeline inspired by Quantum‑NAT, but fully classical."""

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor with residual blocks
        self.features = nn.Sequential(
            ResidualBlock(1, 16, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, stride=2),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Flatten and project to 128 dims
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Self‑attention layer (single head for simplicity)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, n_classes),
            nn.BatchNorm1d(n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized logits of shape (batch, n_classes).
        """
        feats = self.features(x)                # (B, 64, 1, 1)
        feats = self.proj(feats)                # (B, 128)
        feats = feats.unsqueeze(1)              # (B, 1, 128) for attention
        attn_out, _ = self.attn(feats, feats, feats)
        attn_out = attn_out.squeeze(1)           # (B, 128)
        attn_out = self.dropout(attn_out)
        out = self.classifier(attn_out)          # (B, n_classes)
        return out


__all__ = ["QFCModel"]
