"""Hybrid self‑attention module that fuses classical attention, convolutional feature extraction, and a small feed‑forward network.

The module intentionally mirrors the interface of the original SelfAttention.py while adding
- a 2‑D convolution (inspired by the quanvolution example) to embed image‑like inputs,
- a fully‑connected layer (FCL) for dimensionality reduction,
- a lightweight regressor (EstimatorQNN) that operates on the attention output.

The method signature is kept identical to the seed:
    run(rotation_params, entangle_params, inputs)

where ``rotation_params`` and ``entangle_params`` are used to compute the query/key matrices in the classical path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["SelfAttentionHybrid"]


class SelfAttentionHybrid:
    def __init__(self, embed_dim: int = 4, n_features: int = 1):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the attention space.
        n_features : int
            Size of the feature vector fed into the final regressor.
        """
        self.embed_dim = embed_dim
        # Convolutional front‑end (classical quanvolution style)
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2)
        # Dimensionality reduction via a fully connected layer
        self.fcl = nn.Linear(embed_dim, embed_dim)
        # Small regression head (mirrors EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass through the hybrid classical pipeline.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(3 * embed_dim,)`` – used to build query matrix.
        entangle_params : np.ndarray
            Shape ``(embed_dim,)`` – used to build key matrix.
        inputs : np.ndarray
            4‑D tensor of shape ``(B, 1, H, W)`` (image‑like).

        Returns
        -------
        np.ndarray
            Predicted scalar per sample, shape ``(B, 1)``.
        """
        # 1. Convolution + flatten
        x = self.conv(torch.as_tensor(inputs, dtype=torch.float32))
        x = x.view(inputs.shape[0], -1)
        # 2. Classical self‑attention
        q = torch.matmul(x, torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32))
        k = torch.matmul(x, torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32))
        scores = F.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        value = x
        attn_out = torch.matmul(scores, value)
        # 3. Dimensionality reduction
        attn_feat = self.fcl(attn_out)
        # 4. Regression head
        out = self.regressor(attn_feat)
        return out.detach().numpy()
