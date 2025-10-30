import torch
from torch import nn
import numpy as np

class ConvSelfAttentionLayer(nn.Module):
    """Hybrid classical layer combining a convolutional filter with a self‑attention block."""
    def __init__(self, conv_kernel: int = 2, attention_dim: int = 4, threshold: float = 0.0):
        super().__init__()
        # Convolution stage
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True)
        self.threshold = threshold
        # Self‑attention stage
        self.attention_dim = attention_dim
        self.rotation = nn.Parameter(torch.randn(attention_dim, attention_dim))
        self.entangle = nn.Parameter(torch.randn(attention_dim, attention_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, H, W) representing a single‑channel image.
        Returns:
            Tensor of shape (batch, 1) – combined conv‑attention output.
        """
        # Convolution
        conv_out = self.conv(x)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        conv_mean = conv_out.mean(dim=[2, 3])  # (batch, 1)

        # Prepare sequence for attention (flatten spatial dims)
        seq = x.view(x.size(0), -1, 1)  # (batch, seq_len, embed_dim=1)

        # Self‑attention
        query = seq @ self.rotation
        key = seq @ self.entangle
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.attention_dim), dim=-1)
        attn_out = scores @ seq  # (batch, seq_len, embed_dim)

        # Reduce attention output to a single scalar per batch item
        attn_mean = attn_out.mean(dim=[1, 2], keepdim=True)  # (batch, 1)

        # Combine stages multiplicatively
        return conv_mean * attn_mean

__all__ = ["ConvSelfAttentionLayer"]
