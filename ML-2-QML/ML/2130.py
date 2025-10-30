"""Enhanced classical convolutional filter with multi‑channel and attention support."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that can process
    multi‑channel images and optionally learn a threshold/attention
    mask.  The API is deliberately compatible with the seed by
    exposing a ``run`` method that accepts a NumPy array or torch
    tensor and returns a single scalar.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        threshold: float | None = None,
        attention: bool = False,
        bias: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolution kernel. Default is 2.
        in_channels : int
            Number of input channels. Default is 1.
        out_channels : int
            Number of output channels. Default is 1.
        threshold : float | None
            Threshold for the sigmoid activation. If None, 0.0 is used.
        attention : bool
            If True, a channel‑wise attention map is learned.
        bias : bool
            Whether to include a bias term in the convolution.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = 0.0 if threshold is None else threshold
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=bias
        )
        if attention:
            self.attn = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, bias=False
            )
            self.attn_activation = nn.Sigmoid()
        else:
            self.attn = None

    def run(self, data) -> float:
        """
        Run the filter on the input data.

        Parameters
        ----------
        data : numpy.ndarray | torch.Tensor
            Input array of shape (in_channels, H, W) or a 2‑D array
            for single‑channel data.

        Returns
        -------
        float
            Mean activation value after the convolution and sigmoid
            thresholding.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            data = data.float()
        else:
            raise TypeError("data must be a NumPy array or torch Tensor")

        if data.ndim == 2:
            data = data.unsqueeze(0)  # shape (1, H, W)

        if self.attn is not None:
            att_map = self.attn_activation(self.attn(data))
            data = data * att_map

        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

__all__ = ["ConvEnhanced"]
