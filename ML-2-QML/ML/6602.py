"""Enhanced classical model with residuals, attention, and quantum‑aware bottleneck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Simple residual block with two conv layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
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
        return self.relu(out)


class SelfAttention(nn.Module):
    """Simple multi‑head self‑attention for 2‑D feature maps."""
    def __init__(self, in_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_dim // num_heads) ** -0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # B, N, C
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = [t.permute(0, 2, 1) for t in qkv]  # B, heads, N, head_dim
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # B, heads, N, head_dim
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class QuantumNATEnhanced(nn.Module):
    """Hybrid model with residual blocks, self‑attention, and a quantum‑aware bottleneck."""
    def __init__(self, num_classes: int = 4, use_quantum_bottleneck: bool = False):
        super().__init__()
        # Feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        # Self‑attention
        self.attention = SelfAttention(128)
        # Bottleneck placeholder
        self.bottleneck = nn.Linear(128 * 7 * 7, 256)
        # Optional quantum‑aware bottleneck (placeholder)
        self.use_quantum_bottleneck = use_quantum_bottleneck
        if self.use_quantum_bottleneck:
            # In practice, replace with a quantum module.
            self.bottleneck = nn.Linear(128 * 7 * 7, 256)
        # Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attention(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return self.norm(x)


__all__ = ["QuantumNATEnhanced"]
