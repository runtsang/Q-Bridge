"""ConvEnhanced: a classical convolutional filter with depthwise separable layers and optional batch‑norm."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that adds depthwise separable
    convolution, learnable bias, and optional batch‑norm.  The module exposes a
    `run` method that returns a scalar activation (the mean of the sigmoid‑activated
    logits) and a `logits` property that exposes the raw logits for further
    processing.

    The architecture is intentionally kept lightweight so it can be used in
    either training or inference mode.  It can be wrapped in a larger CNN
    or used as a standalone feature extractor.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depthwise: bool = True,
        bias: bool = True,
        norm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel.
        threshold : float
            Activation threshold applied before the sigmoid.
        depthwise : bool
            Use depthwise‑separable convolution if True.
        bias : bool
            Include bias term in convolutions.
        norm : bool
            Apply batch‑normalisation after convolution.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.depthwise = depthwise
        self.bias = bias
        self.norm = norm

        if depthwise:
            # depthwise conv followed by 1×1 point‑wise conv
            self.depthwise_conv = nn.Conv2d(
                1, 1, kernel_size=kernel_size, groups=1, bias=bias
            )
            self.pointwise_conv = nn.Conv2d(1, 1, kernel_size=1, bias=bias)
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

        self.bn = nn.BatchNorm2d(1) if norm else None
        self.sigmoid = nn.Sigmoid()

    def run(self, data) -> float:
        """
        Forward pass that returns a scalar activation.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation after sigmoid.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)

        if self.depthwise:
            logits = self.depthwise_conv(tensor)
            logits = self.pointwise_conv(logits)
        else:
            logits = self.conv(tensor)

        if self.bn is not None:
            logits = self.bn(logits)

        self._last_logits = logits
        activations = self.sigmoid(logits - self.threshold)
        return activations.mean().item()

    @property
    def logits(self):
        """Return the most recent raw logits."""
        if hasattr(self, "_last_logits"):
            return self._last_logits
        raise AttributeError("No logits computed yet. Call run first.")

    def forward(self, x):
        return self.run(x)
