import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class EstimatorQNNHybrid(nn.Module):
    # Hybrid classical estimator that combines a CNN, fraud‑detection inspired layers,
    # and a final regression head.
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters] = (),
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.reduce = nn.Linear(16 * 7 * 7, 2)
        self.fraud_module = build_fraud_detection_program(input_params, layers)
        self.final_fc = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)
        flattened = features.view(bsz, -1)
        reduced = self.reduce(flattened)
        out = self.fraud_module(reduced)
        out = self.final_fc(out)
        return out

def EstimatorQNN() -> EstimatorQNNHybrid:
    # Convenience factory matching the original EstimatorQNN name.
    default_params = FraudLayerParameters(
        bs_theta=0.5,
        bs_phi=0.3,
        phases=(0.1, -0.1),
        squeeze_r=(0.2, 0.2),
        squeeze_phi=(0.0, 0.0),
        displacement_r=(0.5, 0.5),
        displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0),
    )
    return EstimatorQNNHybrid(default_params, [])

__all__ = ["EstimatorQNNHybrid", "EstimatorQNN"]
