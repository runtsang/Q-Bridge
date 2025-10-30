"""ConvFilter: classical convolutional filter with advanced features and training support.

This module provides a drop‑in replacement for the original Conv filter.
It supports multi‑channel convolutions, configurable activation,
pooling, dropout, and exposes a ``run`` method that mimics the old API.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable, Optional, Sequence

__all__ = ["ConvFilter"]


class ConvFilter(nn.Module):
    """
    Classical convolution filter with optional activation, pooling and dropout.

    Parameters
    ----------
    kernel_size : int or tuple
        Size of the convolution kernel.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple, optional
        Stride of the convolution.
    padding : int or tuple, optional
        Zero‑padding added to both sides of the input.
    threshold : float, optional
        Threshold applied to the raw convolution output before the activation.
        If ``None`` no thresholding is performed.
    activation : str or Callable, optional
        Activation function applied after thresholding.  ``'sigmoid'``,
        ``'relu'`` and ``'tanh'`` are supported as strings.  A custom
        callable can also be supplied.
    pooling : str or None, optional
        Pooling operation applied to the output.  ``'max'`` or ``'avg'`` are
        accepted.  ``None`` disables pooling.
    dropout : float or None, optional
        Dropout probability.  ``None`` disables dropout.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int] = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] | None = None,
        threshold: Optional[float] = None,
        activation: str | Callable[[torch.Tensor], torch.Tensor] | None = "sigmoid",
        pooling: str | None = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.threshold = threshold
        self.activation = self._get_activation(activation)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride or 1,
            padding=padding or 0,
            bias=True,
        )

    @staticmethod
    def _get_activation(name_or_callable):
        if name_or_callable is None:
            return lambda x: x
        if isinstance(name_or_callable, str):
            if name_or_callable.lower() == "sigmoid":
                return torch.sigmoid
            if name_or_callable.lower() == "relu":
                return torch.relu
            if name_or_callable.lower() == "tanh":
                return torch.tanh
            raise ValueError(f"Unsupported activation string: {name_or_callable}")
        if callable(name_or_callable):
            return name_or_callable
        raise TypeError("activation must be a string or callable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.conv(x)
        if self.threshold is not None:
            x = x - self.threshold
        x = self.activation(x)
        if self.pooling == "max":
            x = nn.functional.max_pool2d(x, kernel_size=2)
        elif self.pooling == "avg":
            x = nn.functional.avg_pool2d(x, kernel_size=2)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def run(self, data: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
        """
        Public API that mimics the original ``Conv.run`` method.

        Parameters
        ----------
        data : torch.Tensor or nested sequence
            Input data of shape ``(H, W)`` or ``(C, H, W)`` depending on
            ``in_channels``.  If a nested sequence is supplied it is
            converted to a ``torch.Tensor`` with ``dtype=torch.float32``.

        Returns
        -------
        torch.Tensor
            The filter output after applying convolution, activation and
            optional pooling.
        """
        if isinstance(data, (list, tuple)):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            # Add batch and channel dimensions
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(0)
        return self.forward(data)

    def mean_activation(self, output: torch.Tensor) -> float:
        """
        Compute the mean of the activation map.

        Parameters
        ----------
        output : torch.Tensor
            Output from :meth:`forward` or :meth:`run`.

        Returns
        -------
        float
            Mean activation value.
        """
        return output.mean().item()
