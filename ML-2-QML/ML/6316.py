"""Enhanced classical convolutional filter with optional auto‑encoding and depth‑wise separable support.

This module extends the original Conv class:
* multi‑channel support (in_channels, out_channels)
* optional depth‑wise separable convolution
* lightweight auto‑encoder for feature extraction
* fit method for supervised regression
"""

from __future__ import annotations

import torch
from torch import nn, optim
import numpy as np

__all__ = ["ConvHybrid"]


class ConvHybrid(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    threshold : float, default 0.0
        Threshold applied after the convolution before activation.
    depthwise : bool, default False
        If True, use a depth‑wise separable convolution.
    autoencoder : bool, default False
        If True, append a small encoder‑decoder pair to the branch.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        threshold: float = 0.0,
        depthwise: bool = False,
        autoencoder: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.autoencoder = autoencoder

        if depthwise:
            # depth‑wise separable: first depth‑wise conv, then point‑wise
            self.depthwise_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=True
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=True
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=True)

        if autoencoder:
            # encoder reduces channel dimension
            self.encoder = nn.Conv2d(out_channels, max(1, out_channels // 2), kernel_size=1, bias=True)
            # decoder restores channel dimension
            self.decoder = nn.Conv2d(max(1, out_channels // 2), out_channels, kernel_size=1, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass for the classical branch."""
        if hasattr(self, "conv"):
            logits = self.conv(data)
        else:
            logits = self.depthwise_conv(data)
            logits = self.pointwise_conv(logits)
        activations = torch.sigmoid(logits - self.threshold)

        if self.autoencoder:
            encoded = torch.relu(self.encoder(activations))
            decoded = torch.relu(self.decoder(encoded))
            return decoded
        return activations

    def run(self, data: np.ndarray) -> float:
        """Convenience wrapper that accepts a NumPy array and returns a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape: (batch, channels, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        out = self.forward(tensor)
        return out.mean().item()

    def fit(
        self,
        train_loader,
        epochs: int = 5,
        lr: float = 1e-3,
        loss_fn=nn.MSELoss(),
    ) -> None:
        """Simple supervised training loop."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
