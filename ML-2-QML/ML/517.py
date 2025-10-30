import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolutional filter with trainable weights."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class AttentionBlock(nn.Module):
    """Multi‑head self‑attention over flattened feature maps."""
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)           # (batch, 1, dim)
        out, _ = self.attn(x, x, x)  # (batch, 1, dim)
        return out.squeeze(1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid network: classical quanvolution + attention + linear head."""
    def __init__(self, in_channels: int = 1, n_classes: int = 10, heads: int = 4):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        self.attn = AttentionBlock(4 * 14 * 14, heads=heads)
        self.linear = nn.Linear(4 * 14 * 14, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        feats = self.attn(feats)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "AttentionBlock", "QuanvolutionClassifier"]
