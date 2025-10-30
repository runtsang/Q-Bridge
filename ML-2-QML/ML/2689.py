"""
FraudDetectionHybrid: Classical backbone + quantum‑inspired expectation head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical backbone – photonic‑style layered network
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single layer of the classical backbone."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))

def _build_layer(params: FraudLayerParameters, clip: bool) -> nn.Module:
    """Create a single linear‑plus‑activation layer that mirrors a photonic layer."""
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Construct a sequential model that first *exactly* matches the
    first layer of the photonic circuit (the “input”‑layer) and
    then runs the rest of the layers with clipping.
    """
    modules = [_build_layer(input_params, clip=False)]
    modules.extend(_build_layer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2. Quantum‑inspired expectation head
# --------------------------------------------------------------------------- #
class QuantumExpectationHead(nn.Module):
    """
    A lightweight, differentiable expectation value that emulates
    the quantum‑expectation layer from the hybrid models.
    The head uses a 1‑qubit rotation and a simple measurement
    (expectation of Z). The parameters are learnable and
    self‑contained – no external backend required.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        # linear mapping to a single rotation angle
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expectation of Z after a Ry(θ) gate:  cos(θ)
        angles = self.linear(inputs).squeeze(-1)
        return torch.cos(2 * angles + self.shift)

# --------------------------------------------------------------------------- #
# 3. Full hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    End‑to‑end model that combines the classical backbone and
    quantum‑inspired head.  The backbone extracts features from
    input pairs; the quantum head refines the decision boundary
    using a learnable rotation.
    """
    def __init__(self, backbone_params: Iterable[FraudLayerParameters],
                 head_shift: float = 0.0) -> None:
        super().__init__()
        # The first layer is expected to be the input layer
        self.backbone = build_fraud_detection_program(
            backbone_params[0], backbone_params[1:]
        )
        self.head = QuantumExpectationHead(self.backbone[-1].out_features,
                                           shift=head_shift)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass – input tensors are kept as separate tensors."""
        x = self.backbone(*inputs)
        out = self.head(x)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["FraudDetectionHybrid", "build_fraud_detection_program",
           "FraudLayerParameters", "QuantumExpectationHead"]
