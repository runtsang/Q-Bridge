from dataclasses import dataclass
from typing import Iterable, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F

@dataclass
class FraudLayerParameters:
    """Fully‑connected layer parameters adapted from the photonic analogue."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class SamplerQNN(nn.Module):
    """Lightweight sampler network that mirrors the Qiskit SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(self.net(x), dim=-1)

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
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
        def forward(self, x: Tensor) -> Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift
    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    clip: bool = True,
) -> nn.Sequential:
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=clip) for layer in layers)
    modules.append(nn.Linear(2, 1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)

class HybridFraudDetection(nn.Module):
    """Classical side of the hybrid fraud‑detection pipeline."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.classical = build_fraud_detection_program(input_params, layers)
        self.sampler = SamplerQNN()
    def forward(self, x: Tensor) -> Tensor:
        y = self.classical(x)
        probs = self.sampler(y)
        return probs
