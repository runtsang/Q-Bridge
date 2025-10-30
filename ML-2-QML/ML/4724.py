from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Parameters that describe one photonic‑style linear block."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.activation(self.linear(inputs))
            return x * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential network that emulates the photonic layers."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionModel(nn.Module):
    """
    Hybrid fraud‑detector that fuses a CNN, a photonic‑style linear stack,
    and a quantum variational block.

    The input is a 1×28×28 gray‑scale image.  The first two axes are processed
    by the photonic network, the rest by the CNN, and their embeddings are
    concatenated before a final sigmoid output.
    """

    def __init__(self, layer_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_features = 16 * 7 * 7
        self.classical = build_fraud_detection_program(
            input_params=next(iter(layer_params)),
            layers=list(layer_params)[1:],
        )
        # quantum_circuit should be injected externally; placeholder for type safety
        self.quantum_circuit = None  # type: ignore[assignment]
        self.fusion = nn.Linear(self.cnn_features + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        cnn_out = self.cnn(x).view(bsz, -1)
        # Classical stream operates on the first two dimensions of the flattened vector
        class_out = self.classical(cnn_out[:, :2])
        if self.quantum_circuit is None:
            quant_out = torch.zeros(bsz, 1, device=x.device)
        else:
            # Expect quantum_circuit to expose an `input_dim` attribute
            quant_out = self.quantum_circuit(cnn_out[:, :self.quantum_circuit.input_dim])
        fused = torch.cat([class_out, quant_out], dim=1)
        return self.sigmoid(self.fusion(fused))

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionModel"]
