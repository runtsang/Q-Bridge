"""Hybrid quanvolutional regressor with classical convolution and linear regression head.

The module extends the original quanvolution example by adding a regression head
instead of a classification head.  The filter operates on 2×2 patches, producing
a flattened feature map that is fed into a multi‑layer perceptron.  The design
mirrors the style of the QuantumRegression example, making the two variants
comparable for scaling studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """2×2 convolution that extracts local patches from a single‑channel image."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (B, C_in, H, W) -> (B, C_out, H', W')
        features = self.conv(x)
        # flatten spatial dimensions and keep channel dimension
        return features.view(x.size(0), -1)


class QuanvolutionRegressor(nn.Module):
    """Classical quanvolutional network for regression."""
    def __init__(self, in_channels: int = 1,
                 hidden_sizes: tuple[int,...] = (32, 16)) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=in_channels)
        # compute number of features: out_channels * 14 * 14 for 28×28 images
        n_features = 4 * 14 * 14
        layers = [nn.Linear(n_features, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU()]
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        return self.net(features).squeeze(-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionRegressor"]
