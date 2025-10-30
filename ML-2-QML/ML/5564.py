"""Hybrid classical convolutional regression module.

This module fuses the classical convolution from Conv.py with the
EstimatorQNN architecture to provide a lightweight regression head
over the extracted features.  It can be used as a drop‑in
replacement for the quantum quanvolution filter in a purely classical
pipeline.

The class exposes a `run` method that accepts a 2D array (kernel-size × kernel-size)
and returns the regression output.
"""

import torch
from torch import nn
import torch.nn.functional as F


class HybridConvEstimator(nn.Module):
    """
    Classical convolution followed by a small fully‑connected
    regression network (inspired by EstimatorQNN).
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Conv layer mimicking the classical filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # Regression head (EstimatorQNN style)
        self.regressor = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply convolution, flatten, then regress.
        """
        # Expect x shape (N, C=1, H, W)
        conv_out = self.conv(x)          # (N, 1, H-k+1, W-k+1)
        flat = conv_out.view(conv_out.shape[0], -1)  # (N, features)
        # Use first feature as input to regressor
        return self.regressor(flat[:, :1])

    def run(self, data) -> float:
        """
        Convenience wrapper that accepts a 2‑D array of shape
        (kernel_size, kernel_size) and returns a scalar.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        return self.forward(tensor).item()


def Conv():
    """Factory returning a HybridConvEstimator instance."""
    return HybridConvEstimator()
