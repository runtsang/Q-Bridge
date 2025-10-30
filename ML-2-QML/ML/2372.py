"""Hybrid classical quanvolution filter with thresholding and optional kernel size.

This module implements a classical convolutional filter that mimics the
behaviour of the original quanvolution example while adding a learnable
threshold and flexible kernel size.  The filter can be used as a drop‑in
replacement for the quantum version in any PyTorch pipeline.

Classes
-------
QuanvolutionFilter : nn.Module
    Convolutional filter with sigmoid thresholding.
QuanvolutionClassifier : nn.Module
    Simple classifier that stacks the filter and a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]


class QuanvolutionFilter(nn.Module):
    """Classical filter that emulates a quantum quanvolution.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 4
        Number of output channels produced by the convolution.
    kernel_size : int, default 2
        Size of the 2‑D convolution kernel.
    stride : int, default 2
        Stride of the convolution.
    threshold : float, default 0.0
        Sigmoid threshold applied to the convolution output.  Values below
        this threshold are suppressed.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True
        )
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply convolution, threshold and flatten.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Flattened feature vector of shape (B, out_channels * H' * W').
        """
        conv_out = self.conv(x)
        # Apply sigmoid thresholding
        activated = torch.sigmoid(conv_out - self.threshold)
        return activated.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the QuanvolutionFilter followed by a linear head.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 4
        Number of output channels produced by the filter.
    kernel_size : int, default 2
        Size of the filter kernel.
    stride : int, default 2
        Stride of the filter.
    threshold : float, default 0.0
        Threshold applied in the filter.
    num_classes : int, default 10
        Number of target classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            threshold=threshold,
        )
        # Compute flattened feature size
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat = self.qfilter(dummy)
        self.linear = nn.Linear(feat.shape[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)
