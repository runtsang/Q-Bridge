"""Hybrid classical convolution module with multi‑kernel support and learnable threshold.

The original Conv filter was a single 2‑D kernel.  ConvGen264 adds:
* A list of kernels, each defined by a kernel_size and a weight matrix.
* A learnable bias that is used to compute a threshold via a sigmoid.
* A `forward` method that can be dropped into a standard PyTorch `nn.Sequential` or `nn.Module`.
* A `predict` helper that returns the mean activation over a batch.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvGen264(nn.Module):
    """Drop‑in replacement for the original Conv class.

    The module accepts a batch of images (N, C, H, W) and applies a
    convolution with a list of kernel sizes.  The kernel weights are
    learnable and a single learnable threshold is applied to the
    convolution output.  The public `Conv()` function returns an
    instance of this class.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2]
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    bias=bias,
                )
            )
        # learnable threshold parameter
        self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutions and combine the results.

        Args:
            x: Tensor of shape (N, C, H, W).

        Returns:
            Tensor of shape (N, out_channels, H', W') after applying
            sigmoid with the learnable threshold.
        """
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        # sum outputs from all kernels
        out = torch.stack(outs, dim=0).sum(dim=0)
        # apply sigmoid threshold
        out = torch.sigmoid(out - self.threshold)
        return out

    def predict(self, data: torch.Tensor | list[list[float]]) -> float:
        """Return the mean activation for a single image or batch.

        Args:
            data: 4‑D tensor (N, C, H, W) or a 2‑D list for a single image.

        Returns:
            Mean activation over the batch as a float.
        """
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            data = torch.as_tensor(data, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(data).mean().item()

def Conv() -> ConvGen264:
    """Convenience factory that mimics the original API."""
    return ConvGen264()
