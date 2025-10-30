"""Classical convolutional filter with multi‑kernel support and residual connections.

Drop‑in replacement for the original Conv filter.  Extends the original
functionality by:

* Supporting a list of kernel sizes (e.g. [1, 3, 5]) for richer feature extraction.
* Using depthwise separable convolutions for efficient computation.
* Adding a residual connection when the input and output spatial dimensions agree.
* Exposing a ``forward`` method for use in PyTorch models.
* Providing a ``run`` convenience wrapper that accepts a 2‑D array and returns a scalar
  (the mean activation) – mimicking the original API.

Only the PyTorch dependency is required.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List

class ConvGen342(nn.Module):
    """Hybrid multi‑kernel convolutional filter.

    Parameters
    ----------
    kernel_sizes : Iterable[int] | int
        Kernel sizes to use.  If a single int is provided it is treated as
        a list with one element.
    threshold : float, default 0.0
        Activation threshold for the sigmoid function.
    residual : bool, default True
        If ``True`` and the spatial dimensions of the input and the
        concatenated output match, a residual addition is performed.
    """

    def __init__(
        self,
        kernel_sizes: Iterable[int] | int = 3,
        threshold: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = list(kernel_sizes)
        self.threshold = threshold
        self.residual = residual

        # Build a depthwise separable conv for each kernel size
        self.convs: nn.ModuleList = nn.ModuleList()
        for k in self.kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    # Depthwise
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=k,
                        groups=1,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.ReLU(),
                    # Pointwise
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=1,
                        bias=True,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of grayscale images.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(B, 1, H, W)` with pixel values in `[0, 1]`.

        Returns
        -------
        torch.Tensor
            Feature map of shape `(B, 1, H, W)`, mean activation.
        """
        outs: List[torch.Tensor] = []
        for conv in self.convs:
            out = conv(x)
            # Sigmoid activation with threshold
            out = torch.sigmoid(out - self.threshold)
            outs.append(out)

        # Aggregate responses
        out = torch.stack(outs, dim=0).mean(dim=0)  # (B, 1, H, W)

        # Residual addition if dimensions match
        if self.residual and x.shape == out.shape:
            out = out + x
        return out

    def run(self, data: torch.Tensor | list | tuple) -> float:
        """
        Compatibility wrapper that mimics the original Conv.run API.

        Parameters
        ----------
        data : 2‑D array / tensor
            Patch of shape `(kernel_size, kernel_size)`.

        Returns
        -------
        float
            Mean activation over the patch.
        """
        if isinstance(data, (list, tuple)):
            data = torch.as_tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            data = data.to(torch.float32)
        else:
            raise TypeError("Input must be a list, tuple, or torch.Tensor")

        # Reshape to match forward input shape
        data = data.view(1, 1, *data.shape)
        out = self.forward(data)
        return out.mean().item()

def Conv() -> ConvGen342:
    """Return a drop‑in replacement for the original Conv filter."""
    return ConvGen342(kernel_sizes=[1, 3, 5], threshold=0.0, residual=True)
