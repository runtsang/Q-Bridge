from __future__ import annotations

import torch
from torch import nn

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter that emulates a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))  # scalar per sample

class EstimatorNN(nn.Module):
    """Small fully‑connected regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridConvEstimator(nn.Module):
    """Drop‑in replacement that chains a convolutional filter with a regressor."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.estimator = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale patches with shape (batch, 1, k, k).

        Returns
        -------
        torch.Tensor
            Regression predictions of shape (batch, 1).
        """
        features = self.conv_filter(x)
        return self.estimator(features)

__all__ = ["HybridConvEstimator"]
