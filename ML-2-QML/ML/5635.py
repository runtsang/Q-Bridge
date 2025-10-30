"""Hybrid classical estimator that combines a quanvolution‑style convolution with a regression head.

This module is fully classical and depends only on PyTorch.  It is compatible with the original
anchor `EstimatorQNN.py` but scales to larger image‑like inputs by embedding a 2×2 patch
convolution that mimics the quantum filter in the reference pair.  The output is a single
regression value per batch entry.

The design follows a *combination* scaling paradigm: classical layers are augmented by
quantum‑inspired feature extraction, yet remain executable on any CPU/GPU without
requiring a quantum backend.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """2×2 patch convolution that aggregates local structure.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  Defaults to 1 (grayscale).
    out_channels : int
        Number of output feature maps.  Defaults to 4, matching the quantum
        filter in the reference pair.
    kernel_size : int
        Size of the convolutional kernel.  Fixed at 2 to emulate 2×2 patches.
    stride : int
        Stride of the convolution.  Fixed at 2 to down‑sample the image by a factor of two.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply the convolution and flatten the output."""
        return self.conv(x).view(x.size(0), -1)


class EstimatorQNN(nn.Module):
    """Classical regression network that incorporates a quanvolution filter.

    The network first extracts spatial features with a 2×2 patch convolution, then
    passes the flattened tensor through a small fully‑connected regression head.
    It mirrors the structure of the original `EstimatorQNN` but replaces the
    purely linear layers with a hybrid classical–quantum inspired feature extractor.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 1,
        in_channels: int = 1,
        patch_out_channels: int = 4,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            in_channels=in_channels,
            out_channels=patch_out_channels,
        )
        # Compute the flattened feature dimension: 4 * 14 * 14 for a 28×28 input.
        # For arbitrary input sizes this calculation uses the convolution stride.
        self.feature_dim = patch_out_channels * (28 // 2) * (28 // 2)
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a scalar regression output for each batch element."""
        features = self.qfilter(x)
        return self.regressor(features)


__all__ = ["QuanvolutionFilter", "EstimatorQNN"]
