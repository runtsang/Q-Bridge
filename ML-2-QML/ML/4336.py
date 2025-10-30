"""Hybrid classical convolutional network with self‑attention, estimator, and fully connected layer."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvGen010(nn.Module):
    """
    A modular neural network that emulates a quantum‑inspired architecture.
    It sequentially applies a 2‑D convolution, a self‑attention block,
    a small regression head, and a fully connected output layer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        attention_dim: int = 4,
        estimator_hidden: int = 8,
        fc_out_features: int = 1,
    ) -> None:
        super().__init__()

        # Convolutional filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.conv_threshold = conv_threshold

        # Self‑attention parameters (learnable)
        self.attention_dim = attention_dim
        self.rotation_params = nn.Parameter(
            torch.randn(attention_dim, attention_dim)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(attention_dim, attention_dim)
        )

        # Estimator head
        self.estimator = nn.Sequential(
            nn.Linear(attention_dim, estimator_hidden),
            nn.Tanh(),
            nn.Linear(estimator_hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

        # Fully connected output
        self.fc = nn.Linear(1, fc_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Args:
            x: Tensor of shape (batch, 1, H, W).

        Returns:
            Tensor of shape (batch, fc_out_features).
        """
        # Convolution
        conv_out = torch.sigmoid(self.conv(x) - self.conv_threshold)
        conv_out = conv_out.view(x.size(0), -1)  # flatten to (batch, kernel_size**2)

        # Self‑attention
        query = conv_out @ self.rotation_params
        key = conv_out @ self.entangle_params
        scores = torch.softmax(query @ key.t() / np.sqrt(self.attention_dim), dim=-1)
        attn_out = scores @ conv_out

        # Estimator head
        est_out = self.estimator(attn_out)

        # Fully connected
        out = self.fc(est_out)

        return out

__all__ = ["ConvGen010"]
