"""Classical surrogate for the QuanvolutionHybrid model.

The network applies a 2×2 convolution to extract non‑overlapping patches from a
single‑channel image, then feeds each patch to a classical random‑Fourier
feature map that mimics the measurement statistics of a two‑qubit quantum kernel.
The aggregated features are passed to a linear head that can be used for
classification or regression.  The interface is identical to the quantum
implementation so the same experiment scripts can be swapped between the two.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical surrogate for the QuanvolutionHybrid model.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input image channels.
    out_features : int, default 10
        Number of output classes (ignored when ``regression=True``).
    kernel_size : int, default 2
        Size of the convolution kernel that creates image patches.
    stride : int, default 2
        Stride of the convolution kernel.
    num_wires : int, default 64
        Dimensionality of the surrogate quantum feature space.
        Each patch is mapped to a 2·num_wires‑dimensional vector
        via sin/cos Fourier features.
    regression : bool, default False
        If ``True`` the head outputs a single regression value
        instead of class logits.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 10,
        kernel_size: int = 2,
        stride: int = 2,
        num_wires: int = 64,
        regression: bool = False,
    ) -> None:
        super().__init__()
        self.regression = regression
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=kernel_size, stride=stride)

        # Random Fourier feature parameters (fixed)
        self.num_wires = num_wires
        self.W = nn.Parameter(
            torch.randn(4, num_wires) * math.sqrt(2 / num_wires), requires_grad=False
        )
        self.b = nn.Parameter(
            torch.rand(num_wires) * 2 * math.pi, requires_grad=False
        )

        head_in = 2 * num_wires  # sin + cos
        self.head = nn.Linear(head_in, out_features if not regression else 1)

    def _feature_map(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Classical surrogate for the quantum measurement.

        Parameters
        ----------
        patch : torch.Tensor
            Tensor of shape (B·N, 4) where N is the number of patches.

        Returns
        -------
        torch.Tensor
            Tensor of shape (B·N, 2·num_wires) containing sin/cos
            Fourier features.
        """
        z = patch @ self.W + self.b  # (B·N, num_wires)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for classification or a regression output.
        """
        patches = self.conv(x)  # (B, 4, H', W')
        B, C, H, W = patches.shape
        patches = patches.view(B, C, -1).transpose(1, 2)  # (B, N, 4)
        N = patches.shape[1]
        # Flatten patches for feature map
        flat_patches = patches.reshape(-1, 4)
        features = self._feature_map(flat_patches)  # (B·N, 2·num_wires)
        # Aggregate over patches
        features = features.view(B, N, -1).mean(dim=1)  # (B, 2·num_wires)
        logits = self.head(features)  # (B, out_features) or (B, 1)
        if not self.regression:
            return F.log_softmax(logits, dim=-1)
        return logits


__all__ = ["QuanvolutionHybrid"]
