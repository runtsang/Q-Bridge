"""Enhanced classical model for QuantumNAT with residual blocks and multi‑head self‑attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Two‑layer residual block with optional down‑sampling."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = nn.Sequential()
        if stride!= 1 or in_ch!= out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class MultiHeadSelfAttention(nn.Module):
    """Simple multi‑head self‑attention for 2‑D feature maps."""
    def __init__(self, in_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_dim // num_heads) ** -0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        qkv = self.qkv(x_flat)  # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)  # (B, N, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class QuantumNATEnhanced(nn.Module):
    """
    Classical enhancement of the original Quantum‑NAT architecture.
    Adds:
        * ResidualConvBlock after each pooling stage.
        * Multi‑head self‑attention on the flattened feature map before the final linear head.
        * Optional dropout for regularisation.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualConvBlock(8, 8, stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualConvBlock(16, 16, stride=1),
            nn.MaxPool2d(2),
        )
        self.attention = MultiHeadSelfAttention(16, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)          # (B, 16, 7, 7)
        attn_out = self.attention(features)  # (B, 16, 7, 7)
        flattened = attn_out.view(bsz, -1)   # (B, 16*7*7)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
