"""Hybrid classical estimator combining a convolutional feature extractor and a feed‑forward regressor."""

from __future__ import annotations

import torch
from torch import nn


class ConvFilter(nn.Module):
    """Classical 2‑D convolutional filter emulating a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution, sigmoid activation and return mean activation."""
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2, 3))


class EstimatorNN(nn.Module):
    """Simple fully‑connected regression network."""
    def __init__(self, input_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class HybridEstimator(nn.Module):
    """Drop‑in replacement that chains a ConvFilter with an EstimatorNN."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.feature_extractor = ConvFilter(kernel_size, threshold)
        self.regressor = EstimatorNN(input_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, 1, kernel_size, kernel_size)
        Returns:
            Tensor of shape (batch, 1)
        """
        features = self.feature_extractor(inputs).unsqueeze(-1)
        return self.regressor(features)


def get_hybrid_estimator(kernel_size: int = 2, threshold: float = 0.0) -> HybridEstimator:
    """Convenience factory returning a ready‑to‑use HybridEstimator."""
    return HybridEstimator(kernel_size, threshold)


__all__ = ["ConvFilter", "EstimatorNN", "HybridEstimator", "get_hybrid_estimator"]
