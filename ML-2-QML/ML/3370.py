"""Hybrid classical layer combining fully connected and convolutional operations.

The class accepts a list of parameters for the linear part and a 2‑D kernel
for the convolutional part.  The two contributions are combined by a
learnable weight `alpha` that defaults to 0.5.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

import torch
from torch import nn


class HybridLayer(nn.Module):
    """Drop‑in replacement for a quantum fully connected layer that also
    applies a convolutional filter.  The class accepts a list of
    parameters for the linear part and a separate 2‑D kernel for the
    convolutional part.  The two contributions are combined by a
    learned weight ``alpha`` that defaults to 0.5.
    """

    def __init__(
        self,
        n_features: int = 1,
        kernel_size: int = 2,
        threshold: float = 0.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.conv = nn.Conv2d(
            1, 1, kernel_size=kernel_size, bias=True
        )
        self.threshold = threshold
        self.alpha = alpha

    def run(self, thetas: Iterable[float], data: np.ndarray) -> np.ndarray:
        """Return the weighted sum of a linear expectation and a sigmoid
        convolutional activation.
        """
        # Linear part – mimic quantum expectation using tanh activation
        theta_tensor = torch.as_tensor(
            list(thetas), dtype=torch.float32
        ).view(-1, 1)
        lin_out = torch.tanh(self.linear(theta_tensor)).mean(dim=0)

        # Convolutional part
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(
            1, 1, self.conv.kernel_size, self.conv.kernel_size
        )
        logits = self.conv(tensor)
        conv_out = torch.sigmoid(logits - self.threshold).mean()

        combined = self.alpha * lin_out + (1 - self.alpha) * conv_out
        return combined.detach().numpy()


def FCL() -> HybridLayer:
    """Return an instance of the hybrid layer for backward compatibility."""
    return HybridLayer()


__all__ = ["HybridLayer", "FCL"]
