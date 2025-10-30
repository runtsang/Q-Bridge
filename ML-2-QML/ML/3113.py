"""Hybrid QCNN integrating classical and quantum-inspired convolution.

This module defines QCNNHybridModel that extends the original QCNNModel
by incorporating a learnable convolutional filter inspired by the
quantum quanvolution filter from Conv.py. The filter is implemented
as a 2×2 Conv2d layer with a learnable threshold that gates the
input before it is processed by the remaining QCNN layers.

The public API mirrors the original QCNN module:
- QCNN() returns an instance of QCNNHybridModel.
- QCNNHybrid() is an explicit factory for clarity.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvFilter(nn.Module):
    """2×2 convolutional filter with a learnable threshold.

    The filter is inspired by the quantum quanvolution filter
    from Conv.py. It applies a Conv2d with a single input and
    output channel and then thresholds the logits with a
    learnable bias. The output is a scalar per sample,
    which is used to gate the original input.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: Tensor of shape (batch, 1, kernel_size, kernel_size).

        Returns:
            Tensor of shape (batch, 1) containing the mean activation
            after applying the sigmoid threshold.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2, 3), keepdim=True)


class QCNNHybridModel(nn.Module):
    """A QCNN that gates its input through a learnable ConvFilter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            inputs: Tensor of shape (batch, 8).

        The input is first reshaped into a 2×2 image, passed through
        the ConvFilter to obtain a gating scalar, which is then
        broadcasted and multiplied with the original input.
        The gated input is then processed by the standard QCNN layers.
        """
        # Reshape for the ConvFilter
        reshaped = inputs.view(-1, 1, self.conv_filter.kernel_size,
                               self.conv_filter.kernel_size)
        gating = self.conv_filter(reshaped)  # (batch, 1)
        gated_inputs = inputs * gating  # broadcasting to (batch, 8)

        x = self.feature_map(gated_inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.conv_filter.kernel_size}, threshold={self.conv_filter.threshold})"


def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning a freshly configured QCNNHybridModel."""
    return QCNNHybridModel()


def QCNN() -> QCNNHybridModel:
    """Compatibility alias for the original QCNN factory."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybridModel", "QCNN"]
