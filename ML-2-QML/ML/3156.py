"""Hybrid fully connected + convolutional layer with classical back‑end."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Union, Sequence

__all__ = ["HybridFCLConv", "FCL"]


class HybridFCLConv(nn.Module):
    """
    A drop‑in replacement for the original FCL layer that optionally
    includes a 2‑D convolutional filter.  The forward pass behaves
    like the classic FCL when a sequence of thetas is supplied, or
    like the Conv filter when a 2‑D array is supplied.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features for the linear component.
    kernel_size : int, default=2
        Size of the convolutional kernel (square).
    threshold : float, default=0.0
        Activation threshold for the convolutional filter.
    use_conv : bool, default=True
        Whether to enable the convolutional part.
    """

    def __init__(
        self,
        n_features: int = 1,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.linear = nn.Linear(n_features, 1)
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=True,
            )
            self.threshold = threshold
            self.kernel_size = kernel_size

    def _linear_run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the linear branch."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()

    def _conv_run(self, data: Union[Sequence[Sequence[float]], np.ndarray]) -> float:
        """Run the convolutional branch."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def run(self, inputs: Union[Iterable[float], Sequence[Sequence[float]]]) -> Union[np.ndarray, float]:
        """
        Dispatch to the appropriate branch based on input type.

        Returns
        -------
        np.ndarray or float
            Linear expectation for iterable inputs; convolutional
            activation for 2‑D array inputs.
        """
        if isinstance(inputs, (list, tuple, np.ndarray)) and not self.use_conv:
            return self._linear_run(inputs)
        if isinstance(inputs, (list, tuple, np.ndarray)):
            return self._conv_run(inputs)
        raise TypeError("Unsupported input type for HybridFCLConv.run")

    def __call__(self, inputs: Union[Iterable[float], Sequence[Sequence[float]]]) -> Union[np.ndarray, float]:
        return self.run(inputs)


def FCL() -> HybridFCLConv:
    """
    Factory function mirroring the original API.

    Returns
    -------
    HybridFCLConv
        An instance ready for use with either linear or convolutional data.
    """
    return HybridFCLConv()
