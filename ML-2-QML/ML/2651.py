"""
FraudDetectionHybrid – a classical model that fuses photonic‑style layers with a fully‑connected quantum layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single photonic‑style layer.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class PhotonicLayer(nn.Module):
    """
    A lightweight neural‑network analogue of a photonic layer.
    The layer consists of a linear transform followed by a tanh activation,
    then a scaling/shift that mimics displacement gates.
    """
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out


class FCLayer(nn.Module):
    """
    Learnable fully‑connected layer that mimics the quantum FCL example.
    The layer takes a vector of parameters and returns the mean of a tanh activation.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape: (batch, n_features)
        out = torch.tanh(self.linear(thetas))
        return out.mean(dim=0, keepdim=True)


class FraudDetectionML(nn.Module):
    """
    Hybrid classical fraud‑detection model.
    The network consists of:
        1. A photonic‑style input layer.
        2. One or more photonic‑style hidden layers.
        3. A learnable fully‑connected layer (FCL).
        4. A final linear output layer.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Sequence[FraudLayerParameters],
                 fcl_features: int = 1) -> None:
        super().__init__()
        layers = [PhotonicLayer(input_params, clip=False)]
        layers.extend(PhotonicLayer(p, clip=True) for p in hidden_params)
        self.photonic = nn.Sequential(*layers)
        self.fcl = FCLayer(fcl_features)
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, 2) – raw fraud features.
        thetas: tensor of shape (batch, fcl_features) – parameters fed to the FCL.
        """
        out = self.photonic(x)
        fcl_out = self.fcl(thetas)
        # Concatenate photonic output with FCL result before final linear
        cat = torch.cat([out, fcl_out], dim=-1)
        return self.out(cat)


__all__ = ["FraudLayerParameters", "PhotonicLayer", "FCLayer", "FraudDetectionML"]
