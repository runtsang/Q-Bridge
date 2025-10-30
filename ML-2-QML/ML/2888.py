"""
Hybrid self‑attention module combining classical attention with a convolutional
feature extractor inspired by the QCNN seed.  The class is compatible with the
original SelfAttention interface while providing additional depth.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention network.

    Parameters
    ----------
    embed_dim : int, default 4
        Dimensionality of the input embeddings.  This determines the shape of
        the attention weight matrices.
    conv_sizes : list[int], default [16, 12, 8, 4, 4]
        Sizes of the successive fully‑connected layers that emulate the
        convolution‑pooling blocks of the QCNN seed.
    """

    def __init__(self, embed_dim: int = 4, conv_sizes: list[int] | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Attention weight matrices – one per rotation/entangle pair
        self.query_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_weight   = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Convolution‑style feature extractor (mirrors QCNNModel)
        if conv_sizes is None:
            conv_sizes = [16, 12, 8, 4, 4]
        layers = []
        in_dim = embed_dim
        for out_dim in conv_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        """
        Execute the hybrid attention + convolution pipeline.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, embed_dim).
        rotation_params : torch.Tensor, optional
            Parameters to modulate the query weight matrix.  If None the
            learned weight is used directly.
        entangle_params : torch.Tensor, optional
            Parameters to modulate the key weight matrix.  If None the
            learned weight is used directly.

        Returns
        -------
        torch.Tensor
            Output of the sigmoid‑activated convolution head.
        """
        # ---- Self‑attention block ----
        query = torch.matmul(inputs, rotation_params @ self.query_weight if rotation_params is not None else self.query_weight)
        key   = torch.matmul(inputs, entangle_params @ self.key_weight if entangle_params is not None else self.key_weight)
        scores = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = torch.matmul(scores, inputs)

        # ---- Convolution‑style extraction ----
        conv_out = self.conv(attn_out)
        return torch.sigmoid(conv_out)

def HybridSelfAttentionFactory() -> HybridSelfAttention:
    """
    Factory that mirrors the original SelfAttention() API.
    """
    return HybridSelfAttention(embed_dim=4)

__all__ = ["HybridSelfAttention", "HybridSelfAttentionFactory"]
