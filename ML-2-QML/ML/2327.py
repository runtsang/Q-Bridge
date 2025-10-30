"""HybridLayer combining classical convolution and fully connected layers.

This module provides a drop‑in replacement for the original FCL example,
but now includes a convolutional filter before the linear layer.
The class is fully torch‑based and returns a NumPy array to match the
original interface.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable


class HybridLayer(nn.Module):
    """Classic hybrid layer: Conv2d → Sigmoid → Mean → Linear → Tanh → Mean."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        n_features: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.fc = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float], data: np.ndarray | None = None) -> np.ndarray:
        """
        Run the hybrid network.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters for the linear layer.
        data : np.ndarray, optional
            2‑D array of shape (kernel_size, kernel_size) for the conv filter.
            If omitted, only the linear part is evaluated.

        Returns
        -------
        np.ndarray
            Array containing the conv expectation (if data is provided)
            followed by the linear expectation.
        """
        conv_expectation = None
        if data is not None:
            tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            conv_expectation = activations.mean().item()

        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        linear_out = torch.tanh(self.fc(values)).mean(dim=0)
        linear_expectation = linear_out.detach().numpy()[0]

        if conv_expectation is None:
            return np.array([linear_expectation])
        return np.array([conv_expectation, linear_expectation])


__all__ = ["HybridLayer"]
