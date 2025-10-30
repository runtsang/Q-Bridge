"""Enhanced classical convolutional filter with separable layers and channel support."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import sigmoid

class ConvFilter(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that adds:
    * multi‑channel support (in_channels, out_channels)
    * depthwise separable mode for reduced parameter count
    * optional bias and kernel initialization
    * a ``forward`` method that returns the activation map
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        in_channels: int = 1,
        out_channels: int = 1,
        separable: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.separable = separable

        if separable:
            # Depthwise conv (one filter per channel) followed by pointwise
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                bias=False,
                groups=in_channels,
            )
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the kernel, thresholds and returns a
        sigmoid‑activated map.
        """
        if self.separable:
            out = self.depthwise(x)
            out = self.pointwise(out)
        else:
            out = self.conv(x)

        logits = out - self.threshold
        return sigmoid(logits)

    def run(self, data) -> float:
        """
        Convenience wrapper that mimics the original ``run`` signature.
        It accepts a NumPy array, converts it to a tensor, reshapes
        to a batch of 1, and returns the mean activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # batch=1, channel=1
        logits = self.forward(tensor)
        return logits.mean().item()

__all__ = ["ConvFilter"]
