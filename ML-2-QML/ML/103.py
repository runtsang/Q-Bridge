"""Enhanced classical quanvolutional network with residual connections and optional self‑attention.

This implementation mirrors the original hybrid architecture but replaces the fixed
convolution with a depthwise separable block, adds a self‑attention module,
and includes a residual shortcut.  The network is fully differentiable and
serves as a fast baseline for comparison against the quantum version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2D(nn.Module):
    """Simple 2‑D self‑attention over the channel dimension."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        proj_query = self.query(x.view(B, C, -1))
        proj_key = self.key(x.view(B, C, -1))
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x.view(B, C, -1))
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class QuanvolutionNet(nn.Module):
    """Classical quanvolutional network with optional attention and residual shortcut.

    Parameters
    ----------
    use_attention : bool, optional
        If ``True`` a self‑attention block is inserted after the first
        convolution.  Default is ``False``.
    """
    def __init__(self, use_attention: bool = False) -> None:
        super().__init__()
        # Depthwise separable 2×2 convolution
        self.depthwise = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.pointwise = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.ReLU(inplace=True)

        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention2D(4)

        # Residual shortcut matching the output shape
        self.residual_conv = nn.Conv2d(1, 4, kernel_size=1, stride=2, bias=False)

        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.bn(y)
        y = self.act(y)

        if self.use_attention:
            y = self.attn(y)

        residual = self.residual_conv(x)
        y = y + residual

        flat = y.view(y.size(0), -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)
