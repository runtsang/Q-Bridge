"""Hybrid classical FCL implementation inspired by QCNN and the original FCL example."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Sequence


class FCL(nn.Module):
    """
    A classical fully‑connected layer that mimics convolution and pooling
    operations before a final linear head, mirroring the structure of the
    QCNN helper but using 1‑D convolutions for a 1‑D signal.
    """

    def __init__(self, in_features: int = 8, conv_channels: int = 16) -> None:
        super().__init__()
        # Feature extraction: one‑dimensional convolution followed by activation
        self.feature_map = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        # Convolution‑like fully‑connected blocks
        self.conv1 = nn.Sequential(nn.Linear(conv_channels, conv_channels), nn.Tanh())
        self.pool1 = nn.AdaptiveAvgPool1d(12)
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.AdaptiveAvgPool1d(4)
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Final classification head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid classical FCL.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_features). The tensor is reshaped
            to (batch, 1, in_features) for the convolutional layer.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1) after sigmoid activation.
        """
        x = x.unsqueeze(1)  # add channel dimension
        x = self.feature_map(x)
        x = x.view(x.size(0), -1)  # flatten for linear layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def run(self, inputs: Sequence[float]) -> torch.Tensor:
        """
        Convenience method that mirrors the original FCL API.

        Parameters
        ----------
        inputs : Sequence[float]
            Iterable of input features.

        Returns
        -------
        torch.Tensor
            The output of the forward pass as a 1‑D tensor.
        """
        inp = torch.tensor(inputs, dtype=torch.float32)
        return self.forward(inp)


def FCL() -> FCL:
    """Factory that returns a pre‑configured classical FCL instance."""
    return FCL()


__all__ = ["FCL", "FCL"]
