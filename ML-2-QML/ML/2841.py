"""Hybrid classical model combining CNN feature extraction, photonic-inspired
parameterized layers, and a final fully‑connected head.

The architecture is a direct fusion of the Quantum‑NAT CNN backbone and the
FraudDetection parametric layer construction.  It can be used as a drop‑in
replacement for either of the original seeds while providing richer
representation learning.

The model can be instantiated with an optional list of
FraudLayerParameters to pre‑populate the linear stack; otherwise the
parameters are learned from scratch.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QuantumNATHybrid(nn.Module):
    """Hybrid CNN + parametric linear stack."""
    def __init__(
        self,
        cnn_depth: int = 3,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
        out_dim: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # CNN backbone
        layers = []
        in_ch = 1
        for i in range(cnn_depth):
            out_ch = 8 * (i + 1)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        # Linear stack mirroring FraudDetection
        if fraud_params is not None:
            params = list(fraud_params)
            if not params:
                raise ValueError("fraud_params must contain at least one element")
            self.param_block = build_fraud_detection_program(params[0], params[1:])
        else:
            # default random linear stack
            self.param_block = nn.Sequential(
                nn.Linear(2, 2),
                nn.Tanh(),
                nn.Linear(2, 1),
            )
        # Final head
        self.head = nn.Linear(1, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        # Global average pooling to map to 2‑dim vector
        pooled = torch.mean(features, dim=[2, 3])  # shape (bsz, channels)
        # Reduce to 2 dims for the param block
        pooled = pooled[:, :2]
        out = self.param_block(pooled)
        out = self.head(out)
        out = self.norm(out)
        return self.dropout(out)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuantumNATHybrid"]
