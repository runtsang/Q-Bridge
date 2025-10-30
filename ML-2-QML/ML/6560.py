"""Hybrid classical estimator combining a convolution filter and a
fully‑connected regression network.

The module is fully PyTorch‑compatible and can be dropped into any
training loop.  The ConvFilter is a lightweight 2‑D convolution that
produces a single scalar per sample; this scalar is fed into a small
regressor that outputs the final prediction."""
from __future__ import annotations

import torch
from torch import nn

def ConvFilter(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return a convolutional filter that maps a 2‑D patch to a scalar."""
    class _ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=True,
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            Args:
                data: Tensor of shape (batch, 1, H, W)
            Returns:
                Tensor of shape (batch, 1) containing the mean sigmoid
                activation of the convolution output.
            """
            logits = self.conv(data)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean(dim=(2, 3))

    return _ConvFilter()


def EstimatorQNN(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return an end‑to‑end regression network that uses a ConvFilter."""
    class _EstimatorQNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
            self.regressor = nn.Sequential(
                nn.Linear(1, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Args:
                inputs: Tensor of shape (batch, 1, H, W)
            Returns:
                Tensor of shape (batch, 1) containing the predicted value.
            """
            conv_out = self.conv(inputs)          # (batch, 1)
            return self.regressor(conv_out)

    return _EstimatorQNN()


__all__ = ["EstimatorQNN", "ConvFilter"]
