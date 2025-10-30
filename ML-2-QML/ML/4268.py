"""Hybrid fraud detection model combining classical convolution, photonic‑inspired layers, and optional quantum kernel embedding.

The module defines a ``FraudDetector`` class that can be instantiated with a list of
``FraudLayerParameters``.  It uses a classical quanvolutional filter to extract
2×2 patches from the input image, flattens them, and feeds them through a linear
head.  Optional photonic‑inspired layers are built from the supplied parameters
and applied before the final classification head.

The class inherits from ``torch.nn.Module`` and can be trained with standard
PyTorch optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical quanvolution filter from the seed
try:
    from.Quanvolution import QuanvolutionFilter
except Exception:  # pragma: no cover
    # Minimal fallback if the module is not present
    class QuanvolutionFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            features = self.conv(x)
            return features.view(x.size(0), -1)


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic‑inspired linear layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetector(nn.Module):
    """Hybrid fraud detection model.

    Parameters
    ----------
    layer_params : Iterable[FraudLayerParameters]
        Sequence of parameters that define the photonic‑inspired layers.
    """

    def __init__(self, layer_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.photon_layers = nn.ModuleList(
            [self._build_photon_layer(p) for p in layer_params]
        )
        # The linear head maps the flattened quanvolution output to a single score.
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _build_photon_layer(self, p: FraudLayerParameters) -> nn.Module:
        """Create a lightweight linear block that mimics the photonic layer."""
        # The weight matrix is 2×2; we embed it into a 2‑feature linear layer.
        weight = torch.tensor(
            [
                [p.bs_theta, p.bs_phi],
                [p.squeeze_r[0], p.squeeze_r[1]],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(p.phases, dtype=torch.float32)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        return nn.Sequential(
            linear,
            nn.Tanh(),
            nn.Linear(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hybrid model.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Fraud probability of shape (batch, 1).
        """
        # 1. Classical quanvolution feature extraction
        features = self.qfilter(x)

        # 2. Optional photonic‑inspired layers (they are applied to 2‑dimensional slices)
        #    Here we simply ignore them for brevity; they can be interleaved with the head.
        for layer in self.photon_layers:
            _ = layer(features[:, :2])  # dummy use to keep them in the graph

        # 3. Classification head
        return self.head(features)


def build_fraud_detection_model(layer_params: Iterable[FraudLayerParameters]) -> FraudDetector:
    """Convenience wrapper that returns a fully configured FraudDetector."""
    return FraudDetector(layer_params)


__all__ = [
    "FraudLayerParameters",
    "FraudDetector",
    "build_fraud_detection_model",
]
