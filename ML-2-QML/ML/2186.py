"""Fraud detection model – classical implementation."""

from __future__ import annotations

import typing
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import random

@dataclass
class FraudLayerParameters:
    """Parameters of a single dense layer in the fraud‑detection network."""
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

class FraudDetectionModel(nn.Module):
    """
    A modular neural network that mirrors the layered structure of the
    photonic circuit.  Each layer is a linear → tanh → scaling/shift block.
    The first layer can have unbounded weights; subsequent layers are clipped
    to keep the parameters in a realistic regime.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    dropout_rate : float, optional
        Dropout probability applied after every hidden layer.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: typing.Iterable[FraudLayerParameters],
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.layers.append(self._make_layer(input_params, clip=False))
        for params in layers:
            self.layers.append(self._make_layer(params, clip=True))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.out = nn.Linear(2, 1)

    def _make_layer(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
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

            def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
                out = self.activation(self.linear(x))
                return out * self.scale + self.shift

        return Layer()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.out(x)

    @staticmethod
    def synthetic_data(
        n_samples: int,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Generate a toy dataset that mimics the two‑mode photonic inputs.

        Returns
        -------
        X : Tensor of shape (n_samples, 2)
        y : Tensor of shape (n_samples, 1)
        """
        rng = random.Random(seed)
        X = torch.randn(n_samples, 2)
        # A simple linear decision boundary with noise
        y = (X[:, 0] > X[:, 1]).float().unsqueeze(1)
        y += noise_std * torch.randn_like(y)
        y = torch.clamp(y, 0, 1)
        return X, y

__all__ = ["FraudLayerParameters", "FraudDetectionModel", "_clip"]
