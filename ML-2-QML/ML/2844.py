"""Combined classical fraud detection with convolutional preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


# --- Conv filter (from Conv.py) -----------------------------------------

class ConvFilter(nn.Module):
    """Emulates a quantum filter via a 2‑D convolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return a scalar activation."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


# --- Fraud detection layers (from FraudDetection.py) --------------------

@dataclass
class FraudLayerParameters:
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --- Hybrid model -------------------------------------------------------

class FraudDetectorHybrid:
    """
    Classical fraud detector that first applies a convolutional filter
    and then processes the result through a photonic‑inspired linear stack.
    """
    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        self.conv = ConvFilter(conv_kernel_size, conv_threshold)
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.5,
                bs_phi=0.3,
                phases=(0.1, -0.1),
                squeeze_r=(0.2, 0.2),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layers is None:
            fraud_layers = []

        self.fraud = build_fraud_detection_program(fraud_input_params, fraud_layers)

    def run(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → convolution → flatten → fraud network → prediction.
        """
        conv_out = self.conv(image)
        # The convolution returns a scalar; we expand to match the fraud network input
        x = conv_out.unsqueeze(0).repeat(2)  # shape (2,)
        return self.fraud(x)


__all__ = ["FraudDetectorHybrid"]
