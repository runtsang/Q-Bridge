"""
Enhanced classical model for Quantum‑NAT with residual connections and self‑attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers and a skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride!= 1 or in_ch!= out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class SelfAttention(nn.Module):
    """Channel‑wise self‑attention over feature maps."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // 8, kernel_size=1)
        self.key = nn.Conv2d(in_ch, in_ch // 8, kernel_size=1)
        self.value = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, H, W = x.shape
        q = self.query(x).view(batch, -1, H * W)
        k = self.key(x).view(batch, -1, H * W)
        v = self.value(x).view(batch, -1, H * W)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (C ** 0.5), dim=2)
        out = torch.bmm(v, attn.transpose(1, 2)).view(batch, C, H, W)
        return self.gamma * out + x


class QuantumNatModel(nn.Module):
    """Hybrid classical‑quantum model for Quantum‑NAT tasks."""

    def __init__(self) -> None:
        super().__init__()
        # CNN backbone with residuals and attention
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8, 16, stride=2),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, stride=2),
            nn.MaxPool2d(2),
            SelfAttention(32),
        )
        # Flatten and fully connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["QuantumNatModel"]
