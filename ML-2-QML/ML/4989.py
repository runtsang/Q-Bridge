from __future__ import annotations

import torch
from torch import nn


class ConvGen256(nn.Module):
    """
    Classical multi‑scale convolutional filter that can process
    kernels up to 256×256 pixels.  The module is drop‑in compatible
    with the original Conv filter from Conv.py.
    """
    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2, 4, 8, 16, 32, 64, 128, 256]
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        # Create a convolution for each kernel size
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    1,
                    kernel_size=ks,
                    stride=self.stride,
                    padding=self.padding,
                )
                for ks in self.kernel_sizes
            ]
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: Tensor of shape (H, W) or (1, 1, H, W).

        Returns:
            Tensor containing the mean activation over all kernels,
            shape (1,).
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(0)
        activations = []
        for conv in self.convs:
            out = conv(data)
            out = torch.sigmoid(out - self.threshold)
            activations.append(out.mean())
        return torch.stack(activations).mean().view(-1)
