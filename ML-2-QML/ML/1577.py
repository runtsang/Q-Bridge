"""Enhanced classical convolutional filter with depthwise separable support and learnable threshold."""

from __future__ import annotations

import torch
from torch import nn


def Conv():
    """Return a module that emulates the quantum filter with a richer architecture."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0, depthwise: bool = True) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.depthwise = depthwise

            if depthwise:
                # Depthwise separable: one filter per channel
                self.depthwise_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
                self.pointwise_conv = nn.Conv2d(1, 1, kernel_size=1,
                                                bias=False, stride=1, padding=0)
            else:
                self.pointwise_conv = nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=kernel_size,
                    bias=True)

            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
            self.activation = torch.sigmoid

        def forward(self, x):
            """
            Forward pass.

            Args:
                x (torch.Tensor or np.ndarray): Input image of shape
                    (batch, 1, kernel_size, kernel_size).

            Returns:
                torch.Tensor: Mean activation value for each sample in the batch.
            """
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)

            if self.depthwise:
                x = self.depthwise_conv(x)
                x = self.pointwise_conv(x)
            else:
                x = self.pointwise_conv(x)

            logits = x - self.threshold
            activations = self.activation(logits)
            return activations.mean(dim=[1, 2, 3])  # mean per sample

        def run(self, data) -> float:
            """
            Convenience wrapper matching the original API.

            Args:
                data (np.ndarray or torch.Tensor): 2D array with shape
                    (kernel_size, kernel_size).

            Returns:
                float: Mean activation value for the single sample.
            """
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data, dtype=torch.float32)
            data = data.reshape(1, 1, self.kernel_size, self.kernel_size)
            return self.forward(data).item()

    return ConvFilter()


__all__ = ["Conv"]
