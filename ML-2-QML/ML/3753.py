"""Classical surrogate for a hybrid quanvolution + estimator network.

This module implements a purely classical surrogate of the quantum
quanvolution used in the original `Quanvolution.py`.  The convolutional
filter extracts non‑overlapping 2×2 patches from a 28×28 image, which
are then mapped to a higher‑dimensional feature space via a random
Fourier feature map.  The resulting feature vector is fed to a small
fully‑connected head that mirrors the estimator structure of the
quantum `EstimatorQNN`.

The design follows the EstimatorQNN architecture by using a linear
regression head, but the feature extraction is entirely classical,
making the module suitable for environments where quantum back‑ends
are unavailable.

The class is intentionally lightweight and can be swapped with the
quantum implementation (`QuanvolutionHybridNet` in `qml_code`) in
experiments that compare classical versus quantum performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuanvolutionHybridNet(nn.Module):
    """
    Classical surrogate for the hybrid quanvolution + estimator network.

    Parameters
    ----------
    input_channels : int
        Number of input channels. Defaults to 1 for MNIST‑style data.
    hidden_dim : int
        Dimensionality of the random Fourier feature map.
    regression : bool
        If True a linear regression head (single output) is used,
        otherwise a classification head with 10 outputs.
    """

    def __init__(self,
                 input_channels: int = 1,
                 hidden_dim: int = 512,
                 regression: bool = True) -> None:
        super().__init__()
        self.regression = regression
        # 2×2 convolution with stride 2 to produce 14×14 feature map
        self.conv = nn.Conv2d(input_channels, 4, kernel_size=2, stride=2)
        # Random Fourier feature map
        self.W = nn.Parameter(torch.randn(4 * 14 * 14, hidden_dim),
                              requires_grad=False)
        self.b = nn.Parameter(2 * np.pi * torch.rand(hidden_dim),
                              requires_grad=False)
        # Regression / classification head
        if regression:
            self.head = nn.Linear(hidden_dim, 1)
        else:
            self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        feat = self.conv(x)  # shape: (batch, 4, 14, 14)
        feat = feat.view(x.size(0), -1)  # flatten
        # Random Fourier mapping
        z = torch.sin(feat @ self.W + self.b)
        # Output
        out = self.head(z)
        if not self.regression:
            out = F.log_softmax(out, dim=-1)
        return out


__all__ = ["QuanvolutionHybridNet"]
