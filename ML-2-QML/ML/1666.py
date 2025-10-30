"""Enhanced convolutional layer supporting multi‑channel input, dropout, and a learnable threshold.

The class is a drop‑in replacement for the original quanvolution filter.
It keeps the same public API: `Conv()` returns an object with a `run` method.
The new implementation supports batch processing, weight‑sharing across channels
and can be trained end‑to‑end with a standard optimiser.

Example
-------
>>> conv = Conv(in_channels=3, out_channels=1, kernel_size=2,
...             threshold=0.0, dropout=0.2)
>>> data = torch.randn(4, 3, 8, 8)  # batch of 4 RGB images
>>> out = conv.run(data)
>>> out.shape
torch.Size([4])
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Union, Tuple
import numpy as np


class _Conv(nn.Module):
    """Multi‑channel convolution with learnable threshold and optional dropout."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int | Tuple[int, int] = 2,
        threshold: float = 0.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=bias, stride=1, padding=0
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def run(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the convolution and return a per‑sample mean activation.

        Parameters
        ----------
        data
            Input array of shape ``(batch, in_channels, H, W)`` or a
            single image of shape ``(in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch,)`` containing the average
            activation for each sample.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if data.dim() == 3:  # single image
            data = data.unsqueeze(0)
        out = self.conv(data)
        out = self.dropout(out)
        out = torch.sigmoid(out - self.threshold)
        # mean over spatial and channel dims
        return out.mean([1, 2, 3])


def Conv(**kwargs) -> _Conv:
    """Return an instance of the enhanced Conv layer.

    The signature matches the seed: ``Conv(kernel_size=2, threshold=0.0)``.
    Additional keyword arguments are forwarded to the underlying :class:`_Conv`.
    """
    return _Conv(**kwargs)


__all__ = ["Conv"]
