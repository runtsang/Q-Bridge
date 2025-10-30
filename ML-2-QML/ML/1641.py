"""Advanced convolutional backbone with multi‑scale depth‑wise separable convolutions and residual pooling."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import avg_pool2d, relu, sigmoid

class AdvancedConv(nn.Module):
    """
    A drop‑in replacement for the original Conv() factory.
    Implements a multi‑scale depth‑wise separable convolution with residual
    connections and average pooling.  Designed for single‑channel 2‑D data.
    """

    def __init__(self, kernel_sizes=(2, 3), threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        # Depth‑wise separable convolutions for each kernel size
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=k,
                padding=k // 2,
                groups=1,  # depth‑wise for single channel
                bias=True,
            )
            self.convs.append(conv)
        # Residual 1×1 projection to keep dimensions consistent
        self.residual = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of 2‑D data.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Mean activation value after sigmoid and thresholding.
        """
        # Residual input
        residual = self.residual(x)
        out = x
        for conv in self.convs:
            out = conv(out)
            out = relu(out)
            out = out + residual  # residual connection
        out = self.avg_pool(out)
        logits = out
        activations = sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2, 3))  # mean over spatial dims

def Conv() -> AdvancedConv:
    """
    Factory function to preserve API compatibility with the original Conv().
    Returns an instance of AdvancedConv with default parameters.
    """
    return AdvancedConv()

__all__ = ["AdvancedConv", "Conv"]
