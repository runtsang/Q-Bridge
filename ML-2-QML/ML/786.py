"""Hybrid classical convolutional module with residual and batch‑norm support."""

from __future__ import annotations

import torch
from torch import nn


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that adds:
    * Multi‑channel support (input/output channels).
    * Residual connection to preserve low‑frequency features.
    * Optional batch‑norm for better training dynamics.
    * A small 3×3 kernel with learnable weights.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        threshold: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual addition and optional batch‑norm.
        The output is passed through a sigmoid thresholding step
        using the trainable threshold parameter.
        """
        out = self.conv(x)
        res = self.residual(x)
        out = out + res
        if self.bn is not None:
            out = self.bn(out)
        # sigmoid activation with a learnable threshold
        out = torch.sigmoid(out - self.threshold)
        return out

    def run(self, data: torch.Tensor | list | np.ndarray) -> float:
        """
        Convenience wrapper that mimics the original Conv.run
        for compatibility with legacy code.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)
        output = self.forward(tensor)
        return output.mean().item()


__all__ = ["ConvEnhanced"]
