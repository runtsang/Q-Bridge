from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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

def build_fraud_detection_qcnn(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered QCNN structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QCNNModel(nn.Module):
    """Classical QCNN inspired network using fraud‑style layers."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.qcnn = build_fraud_detection_qcnn(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.qcnn(x))

def QCNN() -> QCNNModel:
    """Return a default QCNNModel instance with random fraud parameters."""
    default_input = FraudLayerParameters(
        bs_theta=0.5, bs_phi=0.5, phases=(0.1, -0.1),
        squeeze_r=(0.3, 0.3), squeeze_phi=(0.0, 0.0),
        displacement_r=(0.2, 0.2), displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0)
    )
    default_layers = [
        FraudLayerParameters(
            bs_theta=0.4, bs_phi=0.6, phases=(0.2, -0.2),
            squeeze_r=(0.4, 0.4), squeeze_phi=(0.0, 0.0),
            displacement_r=(0.3, 0.3), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        for _ in range(3)
    ]
    return QCNNModel(default_input, default_layers)

__all__ = ["FraudLayerParameters", "QCNNModel", "QCNN"]
