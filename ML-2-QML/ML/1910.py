"""Classical convolutional filter with trainable kernel, multi‑channel support and learnable threshold.

This module mirrors the original `Conv` interface while adding richer features:
* trainable convolutional kernel via `torch.nn.Conv2d`
* optional stride, padding and dilation
* learnable threshold parameter that can be optimized during training
* support for multiple input and output channels

The public function `Conv()` returns an instance of `ConvFilter`, keeping backward compatibility with the seed.

Example usage:

>>> from Conv__gen097 import Conv
>>> conv = Conv()
>>> conv.run(np.random.rand(2,2))
0.6234
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvFilter(nn.Module):
    """
    Drop‑in replacement for the original quanvolution filter with trainable weights.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default is 1.
    out_channels : int
        Number of output channels. Default is 1.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple, optional
        Zero‑padding added to both sides of the input. Default is 0.
    threshold : float, optional
        Initial value of the learnable threshold used in the sigmoid activation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Learnable threshold that shifts the sigmoid activation
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """
        Apply the convolution followed by a sigmoid activation and
        return the mean activation over all spatial positions and channels.

        Parameters
        ----------
        data : array‑like
            Input data with shape ``(H, W)`` for single‑channel or
            ``(C, H, W)`` for multi‑channel data.

        Returns
        -------
        float
            Mean sigmoid activation after convolution.
        """
        # Ensure we have a 4‑D tensor: (N, C, H, W)
        if isinstance(data, np.ndarray):
            tensor = torch.as_tensor(data, dtype=torch.float32)
        else:
            tensor = data.float()
        if tensor.ndim == 2:  # H x W
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # N=1, C=1
        elif tensor.ndim == 3:  # C x H x W
            tensor = tensor.unsqueeze(0)  # N=1
        elif tensor.ndim!= 4:
            raise ValueError(f"Unsupported input shape {tensor.shape}")

        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

def Conv() -> ConvFilter:
    """
    Factory that returns a ``ConvFilter`` instance with default hyper‑parameters.
    The signature matches the original seed so existing code continues to work.
    """
    return ConvFilter()

__all__ = ["Conv"]
