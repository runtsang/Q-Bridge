"""Classical fraud‑detection model with optional fully‑connected quantum head.

This module implements a classical feature extractor inspired by the
photonic fraud‑detection circuit (anchor seed) and a lightweight
fully‑connected layer that can act as a quantum surrogate head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Photonic‑style layer parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParams:
    """Parameters describing a single photonic layer."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParams, *, clip: bool) -> nn.Module:
    """Create a single deterministic layer from photonic parameters."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
) -> nn.Sequential:
    """
    Build a deterministic PyTorch feature extractor mirroring the photonic layout.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 2. Classical surrogate for a quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """
    Classical stand‑in for a fully‑connected quantum layer.
    Mimics the interface of the quantum FCL from the reference pair.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Return the mean of a tanh‑activated linear layer."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0).detach()


# --------------------------------------------------------------------------- #
# 3. Hybrid fraud‑detection model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection model that can optionally use a quantum head.
    The head defaults to a simple linear layer but can be swapped for a
    quantum implementation defined in the QML module.
    """

    def __init__(
        self,
        input_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = build_fraud_detection_program(input_params, layers)
        self.head = head or nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Two‑column tensor with class probabilities.
        """
        features = self.feature_extractor(x)
        logits = self.head(features)
        probs = self.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = [
    "FraudLayerParams",
    "build_fraud_detection_program",
    "FCL",
    "FraudDetectionHybrid",
]
