"""Hybrid sampler network combining convolutional feature extraction and a simple softmax classifier.

This module defines a classical neural network that first applies a 2‑pixel
quanvolution‑style convolution to the input image, then reduces the
feature map to a two‑dimensional embedding and finally outputs
class probabilities via a softmax. The architecture is inspired by
both the original SamplerQNN and the Quanvolution example, and
provides a convenient drop‑in replacement that can be used in
benchmarking against its quantum counterpart.

The class can accept either a flat 2‑dimensional feature vector
or a single‑channel 28×28 image.  In the image case, a 2×2 stride‑2
convolution is applied to produce 4 feature maps (14×14), which are
flattened and linearly mapped to a 2‑dimensional output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler network.

    The network is composed of:
    * A quanvolution‑inspired 2×2 convolution (stride 2) that
      reduces a 28×28 image to 4 feature maps of size 14×14.
    * A linear layer that maps the flattened feature vector to a
      2‑dimensional embedding.
    * A final softmax that yields class probabilities.

    Parameters
    ----------
    input_shape : tuple[int,...] | None
        If provided, the shape of the expected input tensor.
        If None, the network expects a 2‑dimensional feature vector
        of length 2.
    """

    def __init__(self, input_shape: tuple[int,...] | None = None) -> None:
        super().__init__()
        if input_shape is None or input_shape == (2,):
            # Pure feature‑vector mode
            self.use_conv = False
            self.linear = nn.Linear(2, 2)
        else:
            # Image mode – expect (1, 28, 28) or (batch, 1, 28, 28)
            self.use_conv = True
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=2,
                stride=2,
            )
            # 4 feature maps of size 14×14 -> 4*14*14 = 784
            self.linear = nn.Linear(4 * 14 * 14, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        x = self.linear(x)
        return F.softmax(x, dim=-1)


__all__ = ["HybridSamplerQNN"]
