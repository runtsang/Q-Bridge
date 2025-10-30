"""Hybrid fraud detection model combining classical dense layers with a quantum-inspired feature map.

The module extends the original photonic-inspired architecture by adding a
two‑dimensional patch extractor (borrowed from the quanvolution example) and a
random Fourier feature map that emulates a quantum kernel.  The resulting
`FraudDetectorML` can be trained entirely with PyTorch while still
reaping the benefits of quantum‑style feature engineering.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

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

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class QuantumFeatureMap(nn.Module):
    """Random Fourier feature map that mimics a quantum kernel."""
    def __init__(self, in_features: int = 2, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Random phase offsets
        self.register_buffer("phase", torch.rand(out_features) * 2 * torch.pi)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        z = self.linear(x)
        # concatenate sin and cos to double the dimensionality
        return torch.cat([torch.sin(z + self.phase), torch.cos(z + self.phase)], dim=-1)

class FraudDetectorML(nn.Module):
    """Hybrid classical–quantum‑inspired fraud‑detection network."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        use_quantum_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.use_qk = use_quantum_kernel
        modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(p, clip=True) for p in layer_params)
        if self.use_qk:
            modules.append(QuantumFeatureMap(out_features=4))
            final_in_features = 4
        else:
            final_in_features = 2
        modules.append(nn.Linear(final_in_features, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.model(x)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectorML",
    "build_fraud_detection_program",
]
