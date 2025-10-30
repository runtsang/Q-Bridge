"""Enhanced convolution filter with support for multi‑channel, depth‑wise separable, and hybrid quantum inference.

The module exposes a single factory function ``Conv`` that returns a ``ConvFilter`` instance.  The filter
supports configurable kernel size, learnable threshold, optional depth‑wise separable convolution, and an
``use_quantum`` flag that is kept for API compatibility but is not implemented in the classical branch.
The implementation is fully PyTorch‑based and can be dropped into existing pipelines without modification.

Example
-------
>>> import numpy as np
>>> from conv_gen123 import Conv
>>> filt = Conv(kernel_size=3, threshold=0.1, use_depthwise=True, in_channels=1, out_channels=1)
>>> data = np.random.rand(3, 3)
>>> filt.run(data)
0.5123
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Optional

__all__ = ["Conv"]


class ConvFilter(nn.Module):
    """Internal PyTorch implementation of the convolution filter."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_depthwise: bool = False,
        in_channels: int = 1,
        out_channels: int = 1,
        use_quantum: bool = False,
        quantum_backend: Optional[object] = None,
        shots: int = 100,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.use_depthwise = use_depthwise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_quantum = use_quantum
        self.quantum_backend = quantum_backend
        self.shots = shots

        if self.use_depthwise:
            # Depth‑wise separable convolution: first depth‑wise, then point‑wise
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                bias=True,
            )
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=True
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=True
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Standard forward pass used internally."""
        if self.use_depthwise:
            x = self.depthwise(data)
            x = self.pointwise(x)
        else:
            x = self.conv(data)
        # Apply sigmoid and threshold
        return torch.sigmoid(x - self.threshold)

    def run(self, data: Iterable[float]) -> float:
        """Apply the filter to a 2‑D kernel and return the mean activation.

        Parameters
        ----------
        data
            2‑D array or list of length ``kernel_size * kernel_size``.

        Returns
        -------
        float
            Mean activation over the output feature map.
        """
        if self.use_quantum:
            # Placeholder for hybrid quantum inference – not implemented in the classical branch.
            raise NotImplementedError(
                "Quantum inference is not available in the classical implementation."
            )

        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, self.in_channels, self.kernel_size, self.kernel_size)
        logits = self.forward(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


def Conv(
    kernel_size: int = 2,
    threshold: float = 0.0,
    use_depthwise: bool = False,
    in_channels: int = 1,
    out_channels: int = 1,
    use_quantum: bool = False,
    quantum_backend: Optional[object] = None,
    shots: int = 100,
) -> ConvFilter:
    """Factory function returning a convolution filter instance.

    The signature mirrors the original seed while adding optional depth‑wise and quantum flags.
    """
    return ConvFilter(
        kernel_size=kernel_size,
        threshold=threshold,
        use_depthwise=use_depthwise,
        in_channels=in_channels,
        out_channels=out_channels,
        use_quantum=use_quantum,
        quantum_backend=quantum_backend,
        shots=shots,
    )
