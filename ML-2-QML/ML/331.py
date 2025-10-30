"""Enhanced classical convolutional filter with trainable kernel and batch‑norm.

This module defines :class:`ConvEnhanced`, a drop‑in replacement for the
original `Conv` filter.  The kernel is now learnable and the filter can be
integrated into a larger neural network with standard back‑propagation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvEnhanced(nn.Module):
    """
    2‑D convolutional filter with a learnable 2×2 kernel, optional batch‑norm,
    and a sigmoid activation.  The filter accepts a single‑channel image
    of shape ``(batch, 1, kernel_size, kernel_size)`` and returns a scalar
    per sample.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.  The original implementation used
        a fixed 2×2 filter, but the class supports larger kernels.
    threshold : float, default 0.0
        Bias added to the convolution output before the sigmoid.
    use_batch_norm : bool, default True
        Apply a :class:`torch.nn.BatchNorm2d` after the convolution.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        # initialise weights with a small Gaussian
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv.bias, 0.0)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(1)
        else:
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, kernel_size, kernel_size)``.

        Returns
        -------
        torch.Tensor
            Scalar output per sample (shape ``(batch,)``).
        """
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = out + self.threshold
        out = torch.sigmoid(out)
        # return mean over spatial dimensions to match the original scalar
        return out.mean(dim=[1, 2, 3])

    def run(self, data: torch.Tensor | list | tuple | "np.ndarray") -> float:
        """
        Convenience wrapper that accepts a 2‑D array and returns a float.

        Parameters
        ----------
        data : array‑like
            Shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            The scalar output of the filter.
        """
        import numpy as np

        arr = np.asarray(data, dtype=np.float32)
        if arr.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Input shape {arr.shape} does not match kernel size "
                f"{self.kernel_size}×{self.kernel_size}"
            )
        tensor = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self.forward(tensor).item()


__all__ = ["ConvEnhanced"]
