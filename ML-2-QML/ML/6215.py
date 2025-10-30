"""Hybrid convolutional filter with classical and quantum back‑ends for research experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """
    A drop‑in replacement for a quantum quanvolution layer.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square kernel (default 2).
    in_channels : int, optional
        Number of input channels. Defaults to 1.
    out_channels : int, optional
        Number of output channels. Defaults to 1.
    activation : str, optional
        Activation function name ('relu', 'tanh','sigmoid', 'leaky').
    l2_reg : float, optional
        L2 weight‑regularization coefficient for the convolutional kernel.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        activation: str = "relu",
        l2_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.activation = activation
        self.l2_reg = l2_reg

    # --------------------------------------------------------------------- #
    # Helper functions
    # --------------------------------------------------------------------- #
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Map a single‑word activation function name to a tensor‑wise function."""
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "leaky":
            return F.leaky_relu(x, negative_slope=0.01)
        return x

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution followed by the chosen activation."""
        out = self.conv(x)
        out = self._apply_activation(out)
        return out

    # --------------------------------------------------------------------- #
    # Regularization
    # --------------------------------------------------------------------- #
    def l2_loss(self) -> torch.Tensor:
        """Return L2 penalty for the convolutional kernel."""
        return self.l2_reg * torch.sum(self.conv.weight**2)


__all__ = ["ConvFilter"]
