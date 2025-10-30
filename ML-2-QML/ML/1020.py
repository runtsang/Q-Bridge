"""
Hybrid classical convolutional filter that extends the original Conv filter.
Provides a learnable convolution followed by optional dropout and thresholding.
Designed to be drop-in compatible with the original Conv() function.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, Union

class ConvGen316(nn.Module):
    """
    Classical convolutional filter with optional dropout and threshold.
    Parameters
    ----------
    kernel_size : int or Tuple[int, int]
        Size of the convolution kernel. If an int, the kernel is square.
    dropout_rate : float, default 0.0
        Dropout probability applied after the convolution.
    threshold : float, default 0.0
        Threshold applied to the convolution output before sigmoid.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        dropout_rate: float = 0.0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the convolutional filter.
        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (batch, 1, H, W) or (1, H, W).
        Returns
        -------
        torch.Tensor
            Scalar activation value.
        """
        if data.ndim == 3:
            data = data.unsqueeze(0)
        conv_out = self.conv(data)
        conv_out = self.dropout(conv_out)
        logits = conv_out - self.threshold
        activations = torch.sigmoid(logits)
        return activations.mean()

    def run(self, data: torch.Tensor) -> float:
        """
        Compute the filter response for a single 2D array.
        Parameters
        ----------
        data : torch.Tensor or array-like
            2D array of shape (H, W).
        Returns
        -------
        float
            Mean activation value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(tensor).item()

def Conv(kernel_size: Union[int, Tuple[int, int]] = 2,
         dropout_rate: float = 0.0,
         threshold: float = 0.0) -> ConvGen316:
    """Return a ConvGen316 instance."""
    return ConvGen316(kernel_size, dropout_rate, threshold)

__all__ = ["ConvGen316", "Conv"]
