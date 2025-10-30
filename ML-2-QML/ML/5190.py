"""Hybrid QCNN model with a classical convolutional feature extractor
and a fully‑connected stack that emulates the quantum convolution‑pool
pattern.

The architecture is inspired by the classical QCNNModel in QCNN.py,
the Conv filter from Conv.py, and the depth‑controlled layering
from the quantum QCNN ansatz in QCNN.py.

The model can be used as a drop‑in replacement for QCNNModel
while providing a richer feature‑map that mimics a quantum filter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNNGen070(nn.Module):
    """
    Classical hybrid QCNN.

    Attributes
    ----------
    feature_map : nn.Conv2d
        2‑D convolution that acts as a feature extractor.
    threshold : float
        Bias applied before the sigmoid activation.
    head : nn.Sequential
        Fully‑connected stack that mirrors the conv/pool sequence.
    """

    def __init__(
        self,
        input_channels: int = 1,
        kernel_size: int = 2,
        feature_map_threshold: float = 0.0,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]

        # Feature map: 2‑D convolution + sigmoid
        self.feature_map = nn.Conv2d(
            input_channels,
            1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.threshold = feature_map_threshold

        # Fully‑connected stack that mirrors the conv/pool sequence
        layers = []
        in_dim = kernel_size * kernel_size  # after conv map flatten
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) after sigmoid activation.
        """
        fmap = torch.sigmoid(self.feature_map(x) - self.threshold)
        flat = fmap.view(fmap.size(0), -1)
        out = self.head(flat)
        return torch.sigmoid(out)


__all__ = ["QCNNGen070"]
