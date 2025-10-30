import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

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
    dropout_rate: float = 0.0  # New extension for regularisation

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
    dropout = nn.Dropout(p=params.dropout_rate) if params.dropout_rate > 0 else nn.Identity()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = dropout
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = self.dropout(outputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudDetectionModel(nn.Module):
    """
    Classical fraud‑detection network that mirrors the photonic architecture
    but adds dropout and an explicit L2‑regularisation helper.
    """
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        super().__init__()
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def l2_regularization(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.model[0].weight.device)
        for m in self.model:
            if hasattr(m, "weight"):
                reg += torch.norm(m.weight, p=2)
        return reg

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetectionModel:
    """Convenience factory that returns a ready‑to‑train model."""
    return FraudDetectionModel(input_params, layers)

__all__ = ["FraudLayerParameters", "FraudDetectionModel", "build_fraud_detection_program"]
