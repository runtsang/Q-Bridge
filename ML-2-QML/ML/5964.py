"""Classical hybrid estimator that mimics the EstimatorQNN and Quanvolution ideas."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EstimatorQNNHybrid(nn.Module):
    """
    Combines a 2×2 convolutional feature extractor with a lightweight
    fully‑connected regression head.  The convolution mimics the
    patch‑based quanvolution filter; the linear layers perform
    regression on the flattened features.
    """

    def __init__(self) -> None:
        super().__init__()
        # Patch‑based quantum-inspired feature extractor
        self.qconv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        # Classical regression head
        self.regressor = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape (batch, 1).
        """
        features = self.qconv(x)
        features = features.view(x.size(0), -1)
        return self.regressor(features)


__all__ = ["EstimatorQNNHybrid"]
