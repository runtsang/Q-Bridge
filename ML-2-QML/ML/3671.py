from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Shared data structure for fraud‑detection parameters
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Classical layer construction
# --------------------------------------------------------------------------- #
def _layer_from_params(params: FraudLayerParameters, input_dim: int, *, clip: bool) -> nn.Module:
    weight = torch.zeros((input_dim, 2), dtype=torch.float32)
    weight[0] = torch.tensor([params.bs_theta, params.bs_phi], dtype=torch.float32)
    weight[1] = torch.tensor([params.squeeze_r[0], params.squeeze_r[1]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(input_dim, 2)
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

def build_fraud_detection_program_custom(
    input_dim: int,
    params_list: Sequence[FraudLayerParameters],
) -> nn.Sequential:
    modules: list[nn.Module] = [
        _layer_from_params(params_list[0], input_dim, clip=False)
    ]
    for p in params_list[1:]:
        modules.append(_layer_from_params(p, 2, clip=True))
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Classical quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Hybrid network definition
# --------------------------------------------------------------------------- #
class FraudQuantumHybrid(nn.Module):
    """
    Hybrid network that combines a classical quanvolution filter with a
    photonic‑inspired fraud‑detection backbone.
    """
    def __init__(
        self,
        fraud_params: Sequence[FraudLayerParameters],
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Determine feature dimension after quanvolution
        dummy = torch.zeros((1, 1, 28, 28))
        with torch.no_grad():
            feat_dim = self.qfilter(dummy).shape[1]
        self.fraud_backbone = build_fraud_detection_program_custom(
            input_dim=feat_dim,
            params_list=fraud_params,
        )
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        fraud_out = self.fraud_backbone(features)
        logits = self.classifier(fraud_out)
        return logits

__all__ = ["FraudLayerParameters", "build_fraud_detection_program_custom",
           "QuanvolutionFilter", "FraudQuantumHybrid"]
