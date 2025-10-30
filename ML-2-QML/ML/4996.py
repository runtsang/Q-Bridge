import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class SimpleSelfAttention(nn.Module):
    """Low‑overhead dot‑product self‑attention implemented in PyTorch."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.t() / np.sqrt(k.size(-1)), dim=-1)
        return scores @ v

class QuanvolutionHybrid(nn.Module):
    """
    Classical counterpart of the hybrid quanvolution architecture.
    Combines a 2‑D convolution, an RBF kernel mapping,
    and a lightweight self‑attention block before a linear head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        gamma: float = 1.0,
        support_vectors: int = 50,
        attn_dim: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, stride=stride)
        self.support_vectors = nn.Parameter(
            torch.randn(support_vectors, conv_out_channels * 14 * 14), requires_grad=False
        )
        self.gamma = gamma
        self.attn = SimpleSelfAttention(attn_dim)
        self.linear = nn.Linear(support_vectors + conv_out_channels * 14 * 14, num_classes)

    def _kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, d)
        diff = x.unsqueeze(1) - self.support_vectors.unsqueeze(0)  # (bs, k, d)
        dist_sq = (diff.pow(2)).sum(-1)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)                     # (bs, conv_out_channels, 14, 14)
        flat = features.view(features.size(0), -1)  # (bs, d)
        kernel_feat = self._kernel_features(flat)   # (bs, k)
        # use a fixed projection for attention
        rot_mat = torch.randn(flat.size(1), self.attn.query.out_features, device=x.device)
        attn_in = flat @ rot_mat
        attn_feat = self.attn(attn_in)              # (bs, attn_out)
        combined = torch.cat([kernel_feat, attn_feat], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
