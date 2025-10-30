"""Self‑Attention network enriched with QCNN-inspired convolutional layers.

The classical implementation mirrors a transformer block (query, key, value)
and then feeds the attended representation through a small
convolution‑pooling stack adapted from the QCNN example.
The model is fully differentiable and can be trained with standard
PyTorch optimizers.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class SelfAttentionNet(nn.Module):
    """Self‑attention head followed by a shallow QCNN‑style extractor."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Self‑attention linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # QCNN‑inspired feature extractor
        self.feature_map = nn.Sequential(nn.Linear(embed_dim, 16), nn.Tanh())
        self.conv1       = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1       = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2       = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2       = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3       = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head        = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Scalar output per example after sigmoid activation.
        """
        # Self‑attention block
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.softmax(Q @ K.transpose(-1, -2) / self.embed_dim ** 0.5, dim=-1)
        attn_out = scores @ V

        # QCNN style processing
        out = self.feature_map(attn_out)
        out = self.conv1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = torch.sigmoid(self.head(out))
        return out


def SelfAttention() -> SelfAttentionNet:
    """
    Factory that returns a pre‑configured :class:`SelfAttentionNet` instance
    suitable for quick experimentation.
    """
    return SelfAttentionNet(embed_dim=4)


__all__ = ["SelfAttention", "SelfAttentionNet"]
