"""Hybrid classical convolutional filter.

This module extends the original Conv filter by supporting multiple input and
output channels, weight sharing, and a thresholded sigmoid activation.  It
provides a `forward` method suitable for use inside a PyTorch model and a
`run` method that mimics the original API for quick evaluation on a single
kernel-sized patch.
"""

import torch
from torch import nn
import torch.nn.functional as F

class ConvEnhanced(nn.Module):
    """
    A 2‑D convolutional filter with thresholded sigmoid activation.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the convolution kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    stride : int, default 1
        Stride of the convolution.
    padding : int, default 0
        Zero‑padding added to both sides of the input.
    threshold : float, default 0.0
        Value subtracted from the logits before the sigmoid.  This
        emulates the threshold behaviour of the original quantum filter.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Activated feature map of shape (batch, out_channels, H_out, W_out).
        """
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

    def run(self, data: torch.Tensor) -> float:
        """
        Evaluate the filter on a single 2‑D patch.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean value of the sigmoid‑activated output.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

__all__ = ["ConvEnhanced"]
