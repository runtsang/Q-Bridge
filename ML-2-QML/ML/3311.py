"""Hybrid estimator that fuses a classical convolutional front‑end (inspired by quanvolution) with a shallow feed‑forward regression head."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class HybridEstimatorQNN(nn.Module):
    """
    A small 2‑D regression network that first extracts local features with a
    2×2 convolution and then maps them to a single output via a shallow
    feed‑forward head.  The architecture is a direct synthesis of the
    EstimatorQNN feed‑forward regressor and the QuanvolutionFilter
    convolutional block.
    """
    def __init__(self) -> None:
        super().__init__()
        # 1‑channel input → 4 local feature maps
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Flatten 4×14×14 = 784 features (for 28×28 input)
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).  If a lower‑dimensional
            input is supplied, it is reshaped to (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Regressed scalar per sample.
        """
        if x.dim() == 2:
            # Assume shape (batch, 784) → (batch, 1, 28, 28)
            x = x.view(x.size(0), 1, 28, 28)
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        out = self.fc(features)
        return out

def EstimatorQNN() -> nn.Module:
    """Convenience factory that mirrors the original EstimatorQNN API."""
    return HybridEstimatorQNN()

__all__ = ["EstimatorQNN"]
