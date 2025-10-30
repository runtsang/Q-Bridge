"""Enhanced classical convolution filter for hybrid models.

The :class:`ConvFilter` class extends the original design by adding:
* Full support for arbitrary kernel sizes (2‑8) with padding to keep spatial resolution.
* A learnable bias term that can be zero‑initialized and optionally frozen.
* An adaptive threshold that can be set at runtime or learned as a trainable parameter.
* A ``forward`` method that accepts a batch of images and returns the probability
  that each image satisfies the filter.  The method works with PyTorch tensors
  and can be used inside a ``torch.nn.Module`` or as a standalone callable.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import sigmoid

__all__ = ["ConvFilter"]


class ConvFilter(nn.Module):
    """
    A drop‑in replacement for the original ``Conv`` function.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (square).
    bias : bool, default True
        Whether to include a learnable bias term.
    learnable_threshold : bool, default False
        If ``True`` the threshold is a trainable parameter.
    init_threshold : float, default 0.0
        Initial value for the threshold when not learnable.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        bias: bool = True,
        learnable_threshold: bool = False,
        init_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.learnable_threshold = learnable_threshold

        # Convolution with padding to preserve spatial dimensions
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.tensor(init_threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of grayscale images.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, 1, H, W)`` containing pixel values in
            the range ``[0, 255]`` (or any comparable scale).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch,)`` with a probability value in
            ``[0, 1]`` for each image.
        """
        logits = self.conv(x)  # shape (batch, 1, H, W)
        logits = logits.squeeze(1)  # shape (batch, H, W)
        activations = sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2))

    def set_threshold(self, value: float) -> None:
        """Convenience setter for non‑learnable threshold."""
        if not self.learnable_threshold:
            self.threshold.copy_(torch.tensor(value, dtype=torch.float32))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
