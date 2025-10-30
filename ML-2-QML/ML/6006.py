"""Hybrid classical estimator combining CNN and linear layers with optional quantum‑inspired feature map.

The model can be instantiated in three modes:
* ``linear`` – a tiny feed‑forward regressor (original EstimatorQNN).
* ``cnn`` – a compact CNN followed by a fully‑connected head (QFCModel style).
* ``quantum‑inspired`` – the CNN head followed by a sinusoidal feature mapping that mimics a variational quantum circuit.
"""

from __future__ import annotations

import torch
from torch import nn
import math


class EstimatorQNN(nn.Module):
    """A flexible estimator that supports linear, CNN, or quantum‑inspired backbones."""

    def __init__(
        self,
        mode: str = "linear",
        input_dim: int = 2,
        num_classes: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        mode: str
            One of ``linear``, ``cnn``, or ``quantum``.
        input_dim: int
            Dimensionality of the input for the linear mode.
        num_classes: int
            Number of output units.
        use_batchnorm: bool
            Whether to apply BatchNorm to the final output.
        """
        super().__init__()
        self.mode = mode
        self.use_batchnorm = use_batchnorm

        if mode == "linear":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, num_classes),
            )
        elif mode == "cnn":
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )
        elif mode == "quantum":
            # Quantum‑inspired feature map using sine/cosine of input.
            # This mimics the effect of a parameterized rotation on a qubit.
            self.feature_map = nn.Sequential(
                nn.Linear(input_dim, 5),
                nn.Sigmoid(),
                nn.Linear(5, 10),
                nn.Tanh(),
            )
            self.fc = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if use_batchnorm:
            self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.mode == "linear":
            out = self.net(x)
        elif self.mode == "cnn":
            bsz = x.shape[0]
            features = self.features(x)
            flattened = features.view(bsz, -1)
            out = self.fc(flattened)
        elif self.mode == "quantum":
            # Apply the quantum‑inspired feature map.
            mapped = self.feature_map(x)
            out = self.fc(mapped)
        else:
            raise RuntimeError("Unexpected mode during forward pass")

        if self.use_batchnorm:
            out = self.norm(out)
        return out


__all__ = ["EstimatorQNN"]
