"""Classical Self‑Attention with QCNN feature extraction.

The module defines a `SelfAttentionQCNN` that first transforms the input
through a convolution‑pooling stack (mimicking the QCNN structure) and then
applies a multi‑head dot‑product attention.  The design follows the
`SelfAttention` interface from the anchor seed but augments it with
additional depth and expressivity.

Typical usage::

    from SelfAttention__gen023 import SelfAttention
    attn = SelfAttention()
    output = attn.run(inputs=torch.randn(10, 8))

The class is fully compatible with PyTorch training loops and can be
plugged into larger models.

"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionQCNN(nn.Module):
    """
    Classical self‑attention module that first processes the input
    through a QCNN‑style convolution‑pooling stack and then applies
    a multi‑head attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the feature space after the QCNN extractor.
    num_heads : int, default 4
        Number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # QCNN‑style feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, embed_dim), nn.Tanh()
        )

        # Multi‑head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, feat_dim).

        Returns
        -------
        torch.Tensor
            Attention‑enhanced representation of shape (batch, seq_len, embed_dim).
        """
        # Feature extraction
        batch, seq_len, _ = inputs.shape
        flattened = inputs.view(batch * seq_len, -1)
        features = self.feature_extractor(flattened)
        features = features.view(batch, seq_len, self.embed_dim)

        # Self‑attention
        attn_output, _ = self.attn(features, features, features)
        attn_output = self.out_proj(attn_output)
        return attn_output


def SelfAttention() -> SelfAttentionQCNN:
    """
    Factory function matching the original ``SelfAttention`` signature.
    Instantiates a `SelfAttentionQCNN` with a fixed embedding dimension.
    """
    return SelfAttentionQCNN(embed_dim=4)

__all__ = ["SelfAttention"]
