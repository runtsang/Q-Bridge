from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Callable

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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

class QuantumKernelFeature(nn.Module):
    """Wraps a callable that returns a kernel value between two tensors."""
    def __init__(self, kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.kernel_fn = kernel_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        out = torch.zeros(n, n, device=x.device)
        for i in range(n):
            for j in range(n):
                out[i, j] = self.kernel_fn(x[i], x[j])
        return out.reshape(n, -1)

class FraudDetectionModel(nn.Module):
    """Hybrid fraud detection model combining photonic‑style layers and a quantum kernel."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Sequence[FraudLayerParameters],
        kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        final_dim: int = 1,
    ) -> None:
        super().__init__()
        self.base = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(p, clip=True) for p in hidden_params),
            nn.Linear(2, 2),
        )
        self.kernel_layer = QuantumKernelFeature(kernel_fn)
        self.classifier = nn.Linear(
            2 + (len(hidden_params) * 2) + 4, final_dim
        )  # 2 from last linear, 4 from flattened kernel matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        k = self.kernel_layer(h)
        combined = torch.cat([h, k], dim=1)
        return self.classifier(combined)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a sequential module mirroring the photonic fraud detection architecture."""
    seq = [_layer_from_params(input_params, clip=False)]
    seq.extend(_layer_from_params(p, clip=True) for p in layers)
    seq.append(nn.Linear(2, 1))
    return nn.Sequential(*seq)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionModel",
]
