"""Hybrid classical-quantum sampler combining FraudDetection layers with a variational quantum circuit.

The classical fraud‑detection network processes the input and emits a 2‑dimensional feature vector.
A linear mapping turns this vector into four trainable rotation angles that drive a PennyLane
variational circuit.  The circuit produces a 2‑outcome probability distribution via measurement
of the first qubit.  This design marries the expressive power of the classical layers with the
probabilistic inference of a quantum sampler, enabling end‑to‑end gradient flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# Quantum sampler defined in the companion QML module.
from SamplerQNN__gen010_qml import QuantumSampler


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical fraud‑detection network."""
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
    """Create a sequential PyTorch model mirroring the layered fraud‑detection structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class SamplerQNN__gen010(nn.Module):
    """
    Hybrid sampler that combines the classical fraud‑detection network with a variational
    quantum circuit.  The network outputs a 2‑dimensional feature vector; a linear layer
    transforms it into four rotation angles for the quantum sampler.  The sampler returns
    a 2‑outcome probability distribution.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.fraud_net = build_fraud_detection_program(input_params, layers)
        self.weight_mapper = nn.Linear(2, 4)
        self.quantum_sampler = QuantumSampler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical feature extraction
        features = self.fraud_net(x)  # shape: (batch, 2)
        # Map to quantum rotation angles
        weight_params = self.weight_mapper(features)  # shape: (batch, 4)
        # Quantum inference
        probs = self.quantum_sampler(features, weight_params)  # shape: (batch, 2)
        return probs


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "SamplerQNN__gen010",
]
