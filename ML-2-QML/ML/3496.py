"""HybridConv: classical convolutional backbone with optional pooling.

The class implements a CNN that can be configured with a variable number of
convolutional layers, each followed by a ReLU and a stochastic thresholded
sigmoid.  Pooling layers are inserted every two conv layers to reduce spatial
resolution.  The final representation is flattened and passed through a
fully‑connected head that outputs a single logit.  This design mirrors the
structure of the original Conv.py and QCNN.py seeds while providing a
scalable, modular architecture suitable for larger datasets.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Optional


class HybridConv(nn.Module):
    """
    Classical CNN backbone with configurable depth, kernel size and
    thresholding.  Designed to be a drop‑in replacement for the
    quantum quanvolution layers in the original Conv.py.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 16,
        kernel_size: int = 3,
        depth: int = 4,
        threshold: float = 0.0,
        pool_every: int = 2,
        use_fc: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: int
            Number of input feature maps (default 1 for grayscale images).
        out_channels: int
            Number of output channels for the first conv layer.
        kernel_size: int
            Size of the square convolution kernel.
        depth: int
            Total number of convolutional layers.
        threshold: float
            Threshold applied before the sigmoid activation.
        pool_every: int
            Insert a MaxPool2d layer after every *pool_every* convolutional
            layers.
        use_fc: bool
            Whether to append a fully‑connected head.
        """
        super().__init__()
        self.threshold = threshold
        self.pool_every = pool_every
        self.use_fc = use_fc

        layers: list[nn.Module] = []
        in_ch = in_channels
        for i in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            # Thresholded sigmoid layer
            layers.append(
                nn.Sigmoid()
            )  # We will shift by threshold externally via a custom forward
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_ch = out_channels

        self.features = nn.Sequential(*layers)

        if use_fc:
            # Compute feature map size after pooling
            dummy = torch.zeros(1, in_channels, 32, 32)
            out = self.features(dummy)
            _, _, h, w = out.shape
            self.fc = nn.Linear(out.shape[1] * h * w, 1)
        else:
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the conv layers, a global threshold
        shift, and an optional fully‑connected head.
        """
        x = self.features(x)
        # Apply global threshold to all activations
        x = torch.sigmoid(x - self.threshold)
        if self.fc is not None:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return torch.sigmoid(x)

__all__ = ["HybridConv"]
