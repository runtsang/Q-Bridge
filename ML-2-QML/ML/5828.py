"""Hybrid classical model combining Quantum‑NAT and fraud‑detection inspired layers."""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    """Parameter set for a parametric linear layer used in the fraud‑detection style module."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    # Build a parameterised linear layer with scale/shift post‑processing.
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class HybridNATFraudModel(nn.Module):
    """Classical CNN + fraud‑detection style parametric network.

    Combines the feature extractor from Quantum‑NAT with a stack of
    parameterised linear layers inspired by the photonic fraud‑detection
    circuit, yielding a 4‑dimensional output.
    """

    def __init__(
        self,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
        n_fraud_layers: int = 3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

        # Build fraud‑style sub‑network
        if fraud_params is None:
            # Default parameters – simple identity‑like layers
            dummy = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            fraud_params = [dummy] * n_fraud_layers

        self.fraud_seq = nn.Sequential(
            *[_layer_from_params(p, clip=True) for p in fraud_params]
        )
        # Project fraud sub‑network output to 4‑dimensional space
        self.fraud_fc = nn.Linear(2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out_fc = self.fc(flattened)

        # Fraud sub‑network processes the same flattened representation
        # but only the first two channels are used to keep the interface
        # consistent with the original fraud detection implementation.
        fraud_in = torch.zeros(bsz, 2, device=x.device)
        fraud_out = self.fraud_seq(fraud_in)
        out_fraud = self.fraud_fc(fraud_out)

        # Combine the two streams (e.g. by addition) and normalize
        out = out_fc + out_fraud
        return self.norm(out)

__all__ = ["HybridNATFraudModel"]
