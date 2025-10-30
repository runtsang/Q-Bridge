"""Hybrid classical self‑attention with convolutional feature extractor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class HybridSelfAttention(nn.Module):
    """CNN + classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the attention space.
    img_channels : int, optional
        Number of input image channels.  Defaults to 1.
    """

    def __init__(self, embed_dim: int, img_channels: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Feature extractor mimicking the QFCModel convolutional part
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten size after two 2×2 poolings on 28×28 input
        self.flatten_size = 16 * 7 * 7
        # Projection to the attention dimension
        self.proj = nn.Linear(self.flatten_size, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Forward pass of the hybrid attention.

        Parameters
        ----------
        inputs
            Input image tensor of shape (B, C, H, W).
        rotation_params
            Parameters used to build the query matrix, shape
            (embed_dim * 3,).
        entangle_params
            Parameters used to build the key matrix, shape
            (embed_dim - 1,).
        """
        # Feature extraction
        feats = self.features(inputs)
        flattened = feats.view(feats.size(0), -1)
        # Project to the attention space
        x = self.proj(flattened)  # (B, embed_dim)
        # Build query, key matrices using rotation_params and entangle_params
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        flat_inputs = inputs.view(inputs.shape[0], -1)
        query = flat_inputs @ rot
        key = flat_inputs @ ent
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        out = scores @ x
        return self.norm(out)


def SelfAttention() -> HybridSelfAttention:
    """Factory returning a hybrid classical self‑attention module."""
    return HybridSelfAttention(embed_dim=4)


__all__ = ["SelfAttention"]
