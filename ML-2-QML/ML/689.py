"""ConvEnhanced: depth‑wise separable convolution with multi‑kernel support.

This module provides a drop‑in replacement for the original Conv filter.
It supports arbitrary kernel sizes, optional batch processing, and
returns a mean activation value.  The implementation is fully
compatible with PyTorch and can be used in place of the original Conv
filter.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, Union, Iterable

class ConvEnhanced(nn.Module):
    """Depth‑wise separable convolution with trainable weight & bias.

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the convolution kernel.  If an int is passed, a square
        kernel of that size is used.  If a tuple is passed, the
        first element specifies the height and the second the width.
    depthwise_kernels : int, optional
        Number of depth‑wise kernels to apply.  If >1, the input is
        split into that many channels and each channel is convolved
        separately.  The default is 1 (no depth‑wise separation).
    threshold : float, optional
        Threshold used in the sigmoid activation.  The default is 0.0.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        depthwise_kernels: int = 1,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.depthwise_kernels = depthwise_kernels
        self.threshold = threshold

        # Create a separate Conv2d for each depth‑wise kernel
        self.convs = nn.ModuleList()
        for _ in range(depthwise_kernels):
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=True,
            )
            self.convs.append(conv)

    def forward(self, data: Union[torch.Tensor, Iterable[torch.Tensor]]) -> float:
        """Run the convolution on one or many 2‑D inputs.

        Parameters
        ----------
        data : torch.Tensor or iterable of torch.Tensor
            If a single tensor, it must have shape ``(H, W)``.  If an
            iterable, each element must have that shape and the method
            will return the mean activation over the batch.

        Returns
        -------
        float
            Mean activation after applying the sigmoid with the
            configured threshold.
        """
        if isinstance(data, torch.Tensor):
            data = [data]
        activations = []
        for inp in data:
            tensor = inp.clone().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            out = 0.0
            for conv in self.convs:
                out += conv(tensor)
            logits = out
            act = torch.sigmoid(logits - self.threshold)
            activations.append(act.mean().item())
        return sum(activations) / len(activations)

    def run(self, data: Union[torch.Tensor, Iterable[torch.Tensor]]) -> float:
        """Compatibility alias for the original API."""
        return self.forward(data)

__all__ = ["ConvEnhanced"]
