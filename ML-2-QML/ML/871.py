"""Enhanced classical model with residual blocks and multi‑head attention."""
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
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class MultiHeadSelfAttention(nn.Module):
    """Multi‑head self‑attention over flattened patches."""
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output + x)

class QuantumNATEnhanced(nn.Module):
    """Classical CNN with residual blocks, patch embedding, multi‑head attention, and a final classifier."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8, 8),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16, stride=2),
            nn.MaxPool2d(2)
        )
        # Patch embedding: flatten spatial dims into sequence
        self.patch_embed = nn.Linear(16 * 7 * 7, 64)  # embed_dim
        self.attention = MultiHeadSelfAttention(embed_dim=64, num_heads=2, dropout=0.1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28) typical MNIST
        bsz = x.shape[0]
        features = self.features(x)  # (bsz, 16, 7, 7)
        flat = features.view(bsz, -1)  # (bsz, 784)
        embed = self.patch_embed(flat).unsqueeze(1)  # (bsz, 1, 64)
        attn_out = self.attention(embed)  # (bsz, 1, 64)
        attn_out = attn_out.squeeze(1)  # (bsz, 64)
        logits = self.classifier(attn_out)  # (bsz, num_classes)
        return self.norm(logits)

__all__ = ["QuantumNATEnhanced"]
