"""Classical hybrid convolutional network with a quantum‑inspired filter.

The model merges ideas from the original Conv and Quantum‑NAT seeds:
  • A 2‑D convolution that emulates a quantum filter by applying a sigmoid
    activation after a threshold shift.
  • A CNN backbone identical to the QFCModel.
  • A fully‑connected head and batch‑norm for four‑dimensional outputs.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvGen345(nn.Module):
    """
    Classical surrogate for a quantum convolutional neural network.
    Combines a quantum‑inspired convolution filter with a CNN backbone
    and fully‑connected head similar to the Quantum‑NAT architecture.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_out_channels: int = 8,
        conv_stride: int = 1,
        conv_padding: int = 0,
        threshold: float = 0.0,
        fc_hidden: int = 64,
        fc_out: int = 4,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Quantum‑inspired convolution: 2‑D conv + sigmoid activation
        self.quantum_conv = nn.Conv2d(
            1,
            conv_out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=True,
        )
        self.quantum_activation = nn.Sigmoid()

        # CNN backbone: same as QFCModel
        self.features = nn.Sequential(
            nn.Conv2d(conv_out_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_out),
        )
        self.norm = nn.BatchNorm1d(fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, H, W).

        Returns:
            Tensor of shape (batch, 4) after batch‑norm.
        """
        # Apply quantum‑inspired convolution
        q = self.quantum_conv(x)
        q = self.quantum_activation(q - self.threshold)

        # CNN backbone
        features = self.features(q)
        flattened = features.view(features.size(0), -1)

        # Fully‑connected head
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["ConvGen345"]
