"""Hybrid classical fraud‑detection model that outputs parameters for a quantum circuit.

The implementation mirrors the photonic layout from the seed but replaces
photonic primitives with a deep residual neural network.  The output
vector is intended to parameterise a Qiskit variational circuit in the
accompanying quantum module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer (kept for compatibility)."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionHybrid(nn.Module):
    """
    Classical neural network that produces a vector of parameters for a
    variational quantum circuit.  The architecture is a stack of linear
    layers with batch‑norm, ReLU, and dropout, ending with a residual
    shortcut from the first hidden layer to the last.  The final linear
    layer outputs a fixed‑size vector (default 10) that the quantum
    module interprets as rotation angles.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (32, 64),
        dropout: float = 0.1,
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.model = nn.Sequential(*layers)

        # Residual shortcut from first hidden to the last
        self.residual = nn.Linear(hidden_dims[0], hidden_dims[-1])

        self.final = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        res = self.residual(out[:, : self.residual.in_features])
        out = out + res
        return self.final(out)

    def get_quantum_params(self, x: torch.Tensor) -> np.ndarray:
        """
        Return a flattened NumPy array of parameters to feed into the QML circuit.
        """
        with torch.no_grad():
            params = self.forward(x).cpu().numpy()
        return params.flatten()


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
