"""Hybrid classical convolution module with shared threshold and multi‑kernel support.

The module implements a simple 2D convolution layer with a learnable threshold
parameter. It can optionally produce multiple feature maps by specifying a list
of kernel sizes. The output can be used in a standard PyTorch training loop.

Example usage:

    conv = ConvGen154(kernel_sizes=[2,3], threshold=0.5)
    out = conv(torch.randn(1,1,32,32))
"""

import torch
from torch import nn
from typing import List, Union

class ConvGen154(nn.Module):
    """
    Classical convolutional filter that shares a learnable threshold across
    all kernels. The module can be instantiated with multiple kernel sizes
    (e.g. [2,3,5]) which are applied in parallel and concatenated along the
    channel dimension.
    """
    def __init__(
        self,
        kernel_sizes: Union[int, List[int]] = 2,
        threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        # Shared learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        # Create a convolution for each kernel size
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, 1, kernel_size=k, bias=bias)
                for k in self.kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Concatenated feature maps from all kernels.
            Shape: (batch, len(kernel_sizes), H', W')
        """
        feats = []
        for conv in self.convs:
            logits = conv(x)
            acts = torch.sigmoid(logits - self.threshold)
            feats.append(acts)
        return torch.cat(feats, dim=1)

__all__ = ["ConvGen154"]
