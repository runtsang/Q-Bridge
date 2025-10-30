from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class ConvKernel(nn.Module):
    """Light‑weight 2‑D convolution that returns a scalar feature per batch."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: (batch, 1, seq_len, embed_dim)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])

class RBFKernel(nn.Module):
    """Classical RBF kernel used to re‑weight attention scores."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalAttention(nn.Module):
    """Standard scaled dot‑product attention."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class SelfAttentionHybrid(nn.Module):
    """
    Unified self‑attention module that fuses a convolutional pre‑filter,
    a classical attention block, and an RBF kernel re‑weighting.
    """
    def __init__(
        self,
        embed_dim: int = 4,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.pre_filter = ConvKernel(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.attention = ClassicalAttention(embed_dim=embed_dim)
        self.kernel = RBFKernel(gamma=gamma)
        self.project = nn.Linear(embed_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, 4).
        """
        # Convert sequence to a 2‑D patch for the filter
        patched = x.transpose(1, 2).unsqueeze(1)  # (B, 1, seq_len, embed_dim)
        conv_feat = self.pre_filter(patched).squeeze(1)  # (B, seq_len)
        conv_feat = conv_feat.unsqueeze(-1)  # (B, seq_len, 1)

        attn_out = self.attention(x)  # (B, seq_len, embed_dim)
        kernel_vals = self.kernel(attn_out, attn_out)  # (B, seq_len, 1)

        weighted = attn_out * conv_feat * kernel_vals
        pooled = weighted.mean(dim=1)  # (B, embed_dim)
        return self.project(pooled)

__all__ = ["SelfAttentionHybrid"]
