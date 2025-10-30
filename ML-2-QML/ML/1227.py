import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class QuanvolutionFilter(nn.Module):
    """Enhanced classical filter that replaces the single Conv2d with a learnable
    2‑D convolution and a multi‑head self‑attention block over the flattened
    patch embeddings."""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 n_heads: int = 4,
                 dim_head: int = 32):
        super().__init__()
        # Learnable convolution to extract 2×2 patches
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0)
        # Number of patches in a 28×28 image with stride 2
        self.num_patches = (28 // stride) ** 2
        # Attention over patch embeddings
        self.attention = MultiheadAttention(embed_dim=out_channels,
                                            num_heads=n_heads,
                                            batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        features = self.conv(x)                     # (B, 4, 14, 14)
        B, C, H, W = features.shape
        seq = features.view(B, C, H * W).transpose(1, 2)  # (B, seq_len, C)
        attn_out, _ = self.attention(seq, seq, seq)        # (B, seq_len, C)
        # Flatten back to (B, C * seq_len) = (B, 4 * 196) = (B, 784)
        out = attn_out.transpose(1, 2).contiguous().view(B, -1)
        return out

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the enhanced quanvolutional filter followed
    by a linear classification head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
