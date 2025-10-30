"""Extensible 2‑D convolutional filter with learnable parameters and bias regularisation.

Features added:
* Learnable kernel weights and optional bias.
* Configurable activation function (sigmoid, tanh, relu).
* Bias regularisation to avoid trivial solutions.
* Optional threshold to clip the input before convolution, emulating the quantum filter.

The class is drop‑in compatible: `Conv()` returns an instance that accepts a 2‑D array and returns a scalar feature.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvGen133(nn.Module):
    """
    Drop‑in replacement for the original Conv class.
    The forward pass computes a single scalar feature from a 2‑D patch
    and returns it as a Python float.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        bias: bool = True,
        bias_reg: float = 0.0,
        activation: str = "sigmoid",
        threshold: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel.
        bias : bool
            Whether to include a learnable bias term.
        bias_reg : float
            L2 regularisation coefficient for the bias.
        activation : str
            Activation function to apply after convolution.
            Supported: sigmoid, tanh, relu.
        threshold : float | None
            Optional threshold to clip the input before convolution.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.bias_reg = bias_reg
        self.activation_name = activation.lower()
        if self.activation_name not in {"sigmoid", "tanh", "relu"}:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "tanh":
            return torch.tanh(x)
        return F.relu(x)

    def forward(self, data: torch.Tensor | torch.Tensor) -> float:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor or array-like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar feature produced by the filter.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        if self.threshold is not None:
            data = torch.where(data > self.threshold, torch.ones_like(data), torch.zeros_like(data))
        data = data.view(1, 1, self.kernel_size, self.kernel_size)
        out = self.conv(data)
        out = self._apply_activation(out)
        feature = out.mean().item()
        return feature

    def bias_loss(self) -> torch.Tensor:
        """
        Return the bias regularisation loss.
        """
        if self.bias_reg == 0.0 or self.conv.bias is None:
            return torch.tensor(0.0, dtype=self.conv.weight.dtype, device=self.conv.weight.device)
        return self.bias_reg * torch.norm(self.conv.bias, p=2) ** 2


def Conv() -> ConvGen133:
    """Return a ConvGen133 instance (compatible with the original API)."""
    return ConvGen133()
