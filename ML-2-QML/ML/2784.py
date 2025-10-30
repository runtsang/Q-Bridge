import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

class RBFKernel(nn.Module):
    """Classical RBF kernel with optional learnable width."""
    def __init__(self, gamma: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)
        diff = a - b
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

class HybridQuantumKernelFraudDetector(nn.Module):
    """Hybrid model combining an RBF kernel and a classical fraud‑detection network."""
    def __init__(
        self,
        gamma: float = 1.0,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
        learnable_gamma: bool = False,
    ):
        super().__init__()
        self.rbf = RBFKernel(gamma, learnable=learnable_gamma)
        if fraud_params is None:
            # Default shallow network
            fraud_params = [
                FraudLayerParameters(
                    bs_theta=0.0,
                    bs_phi=0.0,
                    phases=(0.0, 0.0),
                    squeeze_r=(0.0, 0.0),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.0, 0.0),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                ),
                FraudLayerParameters(
                    bs_theta=0.0,
                    bs_phi=0.0,
                    phases=(0.0, 0.0),
                    squeeze_r=(0.0, 0.0),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.0, 0.0),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                ),
            ]
        self.fraud_net = build_fraud_detection_program(fraud_params[0], fraud_params[1:])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Returns a tuple (kernel_matrix, fraud_output).
        """
        kernel_mat = self.rbf.kernel_matrix(x, y)
        fraud_out = self.fraud_net(x)
        return kernel_mat, fraud_out

__all__ = [
    "RBFKernel",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridQuantumKernelFraudDetector",
]
