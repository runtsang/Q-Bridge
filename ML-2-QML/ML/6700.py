import torch
from torch import nn
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

class ResidualBlock(nn.Module):
    """Residual block with batch‑norm and dropout."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn(x)
        out = self.linear(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in={self.linear.in_features}, out={self.linear.out_features})"

def _clip_value(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

class FraudDetection__gen557(nn.Module):
    """Hybrid classical fraud detection model with residual blocks."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters],
                 hidden_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.input_layer = _layer_from_params(input_params, clip=False)
        residual_layers = []
        for params in layer_params:
            residual_layers.append(_layer_from_params(params, clip=True))
            residual_layers.append(ResidualBlock(2, 2, dropout))
        self.residuals = nn.Sequential(*residual_layers)
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.residuals(out)
        out = self.classifier(out)
        return torch.sigmoid(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layers={len(self.residuals)})"

__all__ = ["FraudLayerParameters", "FraudDetection__gen557"]
