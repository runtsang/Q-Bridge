"""Hybrid classical convolution filter.

This module defines ConvGen, a drop‑in replacement for the original Conv filter.
It implements a standard 2‑D convolution followed by a differentiable soft‑threshold
and optional pooling. The class is fully differentiable and can be trained
end‑to‑end with PyTorch optimizers.

Typical usage::

    >>> from ConvGen import Conv
    >>> model = Conv(kernel_size=3, threshold=0.1, trainable_threshold=True)
    >>> out = model.run(torch.randn(3,3))
    >>> print(out)
"""

import torch
from torch import nn


class ConvGen(nn.Module):
    """Classical convolution filter with soft‑threshold."""
    def __init__(
        self,
        kernel_size: int = 3,
        threshold: float = 0.0,
        trainable_threshold: bool = False,
        pool: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = pool
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        if trainable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (H, W) or (1, 1, H, W).

        Returns:
            Tensor of shape (1,) containing the mean activation.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        logits = self.conv(x)
        # Soft threshold: sigmoid(logits - threshold)
        if isinstance(self.threshold, torch.Tensor):
            thresh = self.threshold
        else:
            thresh = torch.tensor(self.threshold, device=logits.device, dtype=logits.dtype)
        activations = torch.sigmoid(logits - thresh)
        if self.pool:
            activations = torch.mean(activations, dim=(2, 3))
        return activations.mean()

    def run(self, data):
        """Compatibility wrapper for the original API."""
        return self.forward(torch.as_tensor(data, dtype=torch.float32)).item()


def Conv(*args, **kwargs):
    """Factory that returns a ConvGen instance."""
    return ConvGen(*args, **kwargs)
