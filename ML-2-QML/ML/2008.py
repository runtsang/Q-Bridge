"""Enhanced Quanvolution architecture with depthwise separable filters and attention head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseQuantumFilter(nn.Module):
    """Depthwise separable convolution that mimics a quantum filter."""
    def __init__(self, in_channels: int, out_channels: int = 4, kernel_size: int = 2, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels * out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              groups=in_channels,
                              bias=False)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class SelfAttentionHead(nn.Module):
    """Global self‑attention head that aggregates flattened features."""
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key   = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out   = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x).unsqueeze(1)          # (B,1,D)
        k = self.key(x).unsqueeze(0)            # (1,N,D)
        v = self.value(x)                       # (B,N,D)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5), dim=-1)
        context = torch.matmul(attn, v).squeeze(1)  # (B,D)
        return F.log_softmax(self.out(context), dim=-1)


class Quanvolution__gen209(nn.Module):
    """Classical Quanvolution network with depthwise separable filters and attention head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.filter = DepthwiseQuantumFilter(in_channels, out_channels=4, kernel_size=2, stride=2)
        # 28×28 input with stride 2 → 14×14 patches → 196 patches × 4 features = 784
        self.attention = SelfAttentionHead(feature_dim=196 * 4, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)          # (B, 784)
        logits   = self.attention(features)
        return logits


__all__ = ["Quanvolution__gen209"]
