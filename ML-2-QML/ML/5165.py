from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention module that emulates the interface of the
    original quantum implementation while adding a convolutional feature
    extractor.  The module is fully differentiable and can be inserted
    inside any PyTorch model.
    """

    def __init__(self,
                 embed_dim: int,
                 kernel_size: int = 2,
                 threshold: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Small 2‑D convolution that mimics the quanvolution filter
        self.filter = nn.Conv2d(1, 1,
                                kernel_size=kernel_size,
                                bias=True)
        self.filter.weight.data.normal_(0.0, 0.1)
        self.filter.bias.data.zero_()

        # Linear projections for query/key/value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            Tensor of shape (batch, seq_len, embed_dim)
        rotation_params
            Unused in the classical path but kept for API compatibility.
        entangle_params
            Unused in the classical path but kept for API compatibility.

        Returns
        -------
        Tensor of shape (batch, seq_len, embed_dim)
            Attention‑weighted representation.
        """
        # Apply the convolutional feature map to each token
        B, L, D = inputs.shape
        feat = inputs.view(B * L, 1, 1, D)
        feat = self.filter(feat)
        feat = torch.sigmoid(feat - self.threshold)
        feat = feat.mean([2, 3]).view(B, L, -1)

        # Linear projections
        Q = self.q_proj(feat)
        K = self.k_proj(feat)
        V = self.v_proj(feat)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return out

__all__ = ["HybridSelfAttention"]
