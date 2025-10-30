from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    'Self‑attention over patch features.'
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = k.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        v = v.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out(out)

class AdvancedQuanvolutionFilter(nn.Module):
    'Classical patch extraction and feature computation.'
    def __init__(self, in_channels: int = 1, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 1, 28, 28]
        features = self.conv(x)
        # Flatten patches: [B, 4, 14, 14] -> [B, 196, 4]
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        features = features.reshape(B, H * W, C)  # [B, 196, 4]
        return features

class AdvancedQuanvolutionClassifier(nn.Module):
    'Classifier using the advanced quanvolution filter and attention.'
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = AdvancedQuanvolutionFilter()
        self.attention = SimpleAttention(dim=4, heads=4)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # [B, 196, 4]
        features = self.attention(features)  # [B, 196, 4]
        features = features.reshape(x.size(0), -1)  # [B, 4*196]
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ['AdvancedQuanvolutionFilter', 'AdvancedQuanvolutionClassifier', 'SimpleAttention']
