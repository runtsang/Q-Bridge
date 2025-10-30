"""Enhanced classical quanvolution network with attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Classical quanvolution network that applies a 2×2 convolution followed by a
    multi‑head self‑attention block and a linear classifier.  The attention
    module allows the network to capture long‑range dependencies across the
    14×14 patch grid produced by the 2×2 convolution.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 num_heads: int = 4,
                 head_dim: int = 4,
                 num_classes: int = 10,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attn = nn.MultiheadAttention(embed_dim=out_channels,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 28, 28)
        conv_out = F.relu(self.bn(self.conv(x)))          # (B, out_channels, 14, 14)
        B, C, H, W = conv_out.shape
        seq = conv_out.view(B, C, H * W).transpose(1, 2)   # (B, 14*14, C)
        attn_out, _ = self.attn(seq, seq, seq)            # self‑attention
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, H, W)
        flat = attn_out.view(B, -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
