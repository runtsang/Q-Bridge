"""HybridQuantumConvNet – classical‑only counterpart with attention and multi‑head classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Self‑attention over flattened spatial features."""
    def __init__(self, in_channels: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_channels, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, H, W)
        b, c, h, w = x.shape
        seq = h * w
        x = x.view(b, c, seq).transpose(1, 2)  # (batch, seq, c)
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out + x
        attn_out = self.norm(attn_out)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return attn_out


class QCNet(nn.Module):
    """CNN + attention + classical head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.attn = AttentionBlock(15, heads=1)
        self.fc1 = nn.Linear(15 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.attn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = torch.sigmoid(x)
        return probs


__all__ = ["AttentionBlock", "QCNet"]
