"""
Hybrid quanvolution‑classifier with classical self‑attention.

The module is intentionally lightweight yet fully self‑contained:
* `QuanvolutionFilter` implements a learnable 2×2 convolution that mirrors the original Quanvolution example.
* `SelfAttention` is a drop‑in replacement for the quantum version; it accepts the same rotation and entanglement parameters but operates purely on tensors with PyTorch.
* `QuanvolutionClassifier` ties the two together, applying attention to the flattened patch features before the classification head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical 2×2 convolution applied to 28×28 inputs.
    Returns a flattened feature vector of shape (batch, 4*14*14).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class SelfAttention:
    """
    Classical self‑attention block that mirrors the quantum interface.
    Parameters are passed in but ignored during the forward pass; they are only
    present to keep the signatures identical across ML and QML variants.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class QuanvolutionClassifier(nn.Module):
    """
    End‑to‑end classifier that concatenates a quanvolution filter with a
    classical self‑attention head, followed by a linear output layer.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        self.attention = SelfAttention()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patch features
        features = self.qfilter(x)  # shape: (batch, 4*14*14)
        patches = features.view(-1, 4)  # reshape to (batch*14*14, 4)

        # Random parameters for reproducibility; in practice these would be learned
        rot = np.random.randn(4 * 4)
        ent = np.random.randn(4 * 4)

        # Apply attention
        attended = self.attention.run(rot, ent, patches.numpy())
        attended_tensor = torch.tensor(attended, dtype=torch.float32)

        logits = self.linear(attended_tensor)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "SelfAttention", "QuanvolutionClassifier"]
