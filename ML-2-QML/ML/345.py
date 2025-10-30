"""Hybrid convolution module combining depthwise separable conv and classical quantum simulation.

The ConvHybrid class implements a depthwise separable convolution with a learnable fusion weight that blends the classical convolution output with a classical simulation of a quantum parity measurement. The quantum part is a simple parity-based feature extractor that mimics a quantum measurement of a circuit with one qubit per pixel. The module is fully differentiable and can be trained end‑to‑end using PyTorch optimizers.

Typical usage::

    >>> from Conv__gen383 import ConvHybrid
    >>> model = ConvHybrid(kernel_size=3, num_channels=1, use_quantum=True)
    >>> out = model(torch.randn(1, 1, 28, 28))

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ConvHybrid"]


class ConvHybrid(nn.Module):
    """
    Depthwise separable convolution fused with a classical quantum simulation.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    num_channels : int
        Number of input channels.
    threshold : float
        Threshold used for the quantum parity simulation.
    use_quantum : bool, default=True
        If True, includes the quantum simulation in the forward pass.
    fusion_init : float, default=0.5
        Initial value for the fusion weight between classical and quantum outputs.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        num_channels: int = 1,
        threshold: float = 0.5,
        use_quantum: bool = True,
        fusion_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.threshold = threshold
        self.use_quantum = use_quantum

        # Depthwise separable convolution: depthwise part
        self.depthwise = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            groups=num_channels,
            bias=True,
        )
        # Pointwise part: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=True,
        )

        # Learnable fusion weight (sigmoid constrained to (0,1))
        self.fusion_logit = nn.Parameter(
            torch.tensor(np.log(fusion_init / (1 - fusion_init)), dtype=torch.float32)
        )

    def _quantum_parity(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Classical simulation of a quantum parity measurement.

        Parameters
        ----------
        windows : torch.Tensor
            Tensor of shape (batch, channels, kernel_size*kernel_size, H_out*W_out)
            containing values in [0,1] after thresholding.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, channels, H_out, W_out) with the parity
            probability for each window.
        """
        batch, channels, ksize2, hw = windows.shape
        # Threshold the windows
        binary = (windows > self.threshold).float()
        # Compute parity (sum of ones mod 2)
        parity = binary.sum(dim=2) % 2
        parity = parity.float()
        # Reshape back to spatial dimensions
        H_out = int(np.sqrt(hw))
        W_out = hw // H_out
        return parity.view(batch, channels, H_out, W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after fusion of classical and quantum features.
        """
        # Classical depthwise separable conv
        out_depth = self.depthwise(x)
        out_point = self.pointwise(out_depth)

        if not self.use_quantum:
            return out_point

        # Extract sliding windows for quantum simulation
        # Shape: (batch, channels, kernel_size*kernel_size, H_out*W_out)
        windows = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=0,
            stride=1,
        ).view(x.size(0), self.num_channels, self.kernel_size * self.kernel_size, -1)

        # Compute quantum parity probability
        quantum_out = self._quantum_parity(windows)

        # Fusion weight in (0,1)
        fusion_weight = torch.sigmoid(self.fusion_logit)

        # Fuse classical and quantum outputs
        fused = fusion_weight * out_point + (1 - fusion_weight) * quantum_out

        return fused


def Conv() -> ConvHybrid:
    """Return a hybrid convolution filter."""
    return ConvHybrid(kernel_size=3, num_channels=1, threshold=0.5, use_quantum=True)
