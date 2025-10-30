from __future__ import annotations

import torch
from torch import nn
import numpy as np

class HybridConvEstimator(nn.Module):
    """
    A hybrid classical convolutional regressor that extends the basic Conv filter
    with multi‑kernel support, optional residual connections, and a deeper
    feed‑forward regressor.  It is designed to be a drop‑in replacement for
    the original Conv class while providing richer feature extraction.
    """

    def __init__(
        self,
        kernel_sizes: list[int] = [2, 3],
        threshold: float = 0.0,
        residual: bool = True,
        hidden_dims: list[int] = [16, 8],
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.residual = residual

        # Build a list of conv layers, one per kernel size
        conv_layers = []
        for k in kernel_sizes:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    bias=True,
                )
            )
        self.convs = nn.ModuleList(conv_layers)

        # Build a deeper regressor
        layers = []
        in_features = len(kernel_sizes)
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Tanh())
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Predicted scalar per image.
        """
        # Apply each conv, apply threshold, and average
        conv_outs = []
        for conv in self.convs:
            logits = conv(x)
            act = torch.sigmoid(logits - self.threshold)
            conv_outs.append(act.mean(dim=[2, 3]))  # average over spatial dims

        # Concatenate features
        features = torch.cat(conv_outs, dim=1)

        # Optional residual connection (add back original image mean)
        if self.residual:
            img_mean = x.mean(dim=[2, 3], keepdim=True)
            img_feat = img_mean.view(x.shape[0], -1)
            features = torch.cat([features, img_feat], dim=1)

        return self.regressor(features)

__all__ = ["HybridConvEstimator"]
