"""ConvGen – a depth‑wise separable convolution with a residual gating mechanism.

The module builds on the original single‑kernel filter by:
* Using depth‑wise separable convolutions so each channel is processed independently.
* Adding a lightweight residual branch that normalises the output.
* Providing a trainable `threshold` hyper‑parameter that can be tuned during training.

The class is fully compatible with the original API: `ConvGen()` returns an instance with a `run` method that takes a 2‑D array (or torch tensor) and yields a scalar activation.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Union, Tuple


class ConvGen(nn.Module):
    """Depth‑wise separable convolution with residual gating."""

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        channels: int = 1,
        threshold: float = 0.0,
        residual: bool = True,
        weight_decay: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolutional kernel.
        stride : int
            Stride of the depth‑wise convolution.
        padding : int
            Zero‑padding added to both sides of the input.
        channels : int
            Number of input channels (default 1 for grayscale).
        threshold : float
            Activation threshold for gating the residual branch.
        residual : bool
            Whether to include the residual branch.
        weight_decay : float
            L2 penalty applied to the depth‑wise weights.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.residual = residual

        # Depth‑wise separable convolution: depth‑wise + point‑wise
        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        # Residual branch (identity mapping followed by sigmoid gating)
        if residual:
            self.residual_gate = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid(),
            )

        # Weight decay regulariser
        self.weight_decay = weight_decay

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for a single patch.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input patch of shape (kernel_size, kernel_size) or (1, 1, kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Scalar activation value.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        out = self.depthwise(x)
        out = self.pointwise(out)  # (1, 1, H', W')
        out = torch.sigmoid(out - self.threshold)

        if self.residual:
            # Global average pooling to produce a scalar
            residual = out.mean(dim=[2, 3], keepdim=True)
            gate = self.residual_gate(residual)
            out = out * gate

        return out.mean().item()

    def run(self, data: Union[np.ndarray, torch.Tensor]) -> float:
        """Convenience wrapper to match the original API."""
        return self.forward(data)

    def parameters(self):
        """Return parameters for external optimisers."""
        return super().parameters()


__all__ = ["ConvGen"]
