import torch
import torch.nn as nn
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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
            self.bn = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(0.1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            outputs = self.bn(outputs)
            outputs = self.dropout(outputs)
            return outputs

    return Layer()

class FraudDetectionHybrid(nn.Module):
    """
    A hybrid classical model that mirrors the photonic fraud‑detection structure
    with added regularisation layers.  The network depth is inferred from the
    number of provided layer parameters.
    """
    def __init__(self, depth: int | None = None, clip: bool = True) -> None:
        super().__init__()
        self.clip = clip
        self.layers = nn.ModuleList()
        if depth is None:
            depth = 3
        for _ in range(depth):
            self.layers.append(_layer_from_params(FraudLayerParameters(0,0,(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)), clip=clip))
        self.head = nn.Linear(2, 1)
        if clip:
            nn.init.uniform_(self.head.weight, a=-5.0, b=5.0)
            nn.init.uniform_(self.head.bias, a=-5.0, b=5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.head(x))

    def weight_sizes(self) -> List[int]:
        sizes = []
        for layer in self.layers:
            if isinstance(layer, nn.Module) and hasattr(layer, 'linear'):
                sizes.append(layer.linear.weight.numel() + layer.linear.bias.numel())
        sizes.append(self.head.weight.numel() + self.head.bias.numel())
        return sizes

    @classmethod
    def build_fraud_detection_program(
        cls,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> "FraudDetectionHybrid":
        """
        Construct a FraudDetectionHybrid model from a list of layer parameters.
        The first layer is treated as an input layer and is not clipped.
        """
        model = cls(depth=len(layers) + 1, clip=True)
        # Replace the auto‑generated layers with the ones derived from parameters
        model.layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)]
            + [_layer_from_params(lp, clip=True) for lp in layers]
        )
        return model

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
