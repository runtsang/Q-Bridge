"""Hybrid classical convolutional filter with adaptive threshold and optional learnable bias.

The ConvEnhanced class inherits from torch.nn.Module and implements a
2‑D convolution with an additional learnable threshold that gates the
output.  The class exposes a `run(data)` helper that accepts a 2‑D numpy
array and returns the mean activation value, making it drop‑in
compatible with the original quantvolution interface.

The design keeps the original simplicity (single‑channel input) but
extends it to support multi‑channel data, a learnable bias, and
automatic differentiation via PyTorch.
"""
import torch
from torch import nn
import numpy as np

class ConvEnhanced(nn.Module):
    """
    Classical 2‑D convolutional filter with a learnable threshold.
    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    threshold : float, default 0.0
        Initial value of the learnable threshold.
    learnable : bool, default True
        If True the threshold is a torch Parameter and can be trained.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 threshold: float = 0.0,
                 learnable: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learnable = learnable
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              bias=True)
        # Learnable threshold
        if learnable:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.threshold = torch.tensor(threshold, dtype=torch.float32)
        # Optional bias scaling
        self.bias = nn.Parameter(torch.zeros(out_channels)) if learnable else torch.zeros(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the filter.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, H, W).
        Returns
        -------
        torch.Tensor
            Activated output of shape (batch, out_channels, H', W').
        """
        logits = self.conv(x)  # shape: (batch, out_channels, H', W')
        logits = logits - self.threshold
        logits = logits + self.bias.view(1, -1, 1, 1)
        return torch.sigmoid(logits)

    def run(self, data):
        """
        Convenience wrapper that mimics the original API.
        Accepts a 2‑D numpy array and returns the mean activation.
        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input patch of shape (H, W).
        Returns
        -------
        float
            Mean activation value.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        # Add batch and channel dimensions
        data = data.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        out = self.forward(data)
        return out.mean().item()

__all__ = ["ConvEnhanced"]
