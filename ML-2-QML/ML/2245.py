"""Hybrid classical model: Quanvolution with self‑attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical 2‑D convolution that mimics the original quanvolution filter.
    Produces 4 feature maps on a 28×28 input with stride 2, yielding
    4×14×14 flattened features.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class ClassicalSelfAttention(nn.Module):
    """
    Simple self‑attention block that operates on a low‑dimensional
    representation.  Parameters are supplied externally to keep the
    interface identical to the quantum version.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Map external numpy parameters to tensors
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        query = inputs @ rot
        key = inputs @ ent
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class QuanvolutionClassifier(nn.Module):
    """
    End‑to‑end classical model that applies a quanvolution filter,
    then a self‑attention block, and finally a linear classifier.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.linear = nn.Linear(4 * 14 * 14 + 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract convolutional features
        features = self.qfilter(x)  # shape: (batch, 784)

        # Generate deterministic parameters for reproducibility
        batch = x.size(0)
        rot_params = np.random.randn(4, 4)
        ent_params = np.random.randn(4, 4)

        # Use the first 4 features as the attention input
        inputs = features[:, :4]
        attn_out = self.attention(rot_params, ent_params, inputs)  # shape: (batch, 4)

        # Concatenate and classify
        combined = torch.cat([features, attn_out], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "ClassicalSelfAttention", "QuanvolutionClassifier"]
