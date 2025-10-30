import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridModel(nn.Module):
    def __init__(self,
                 conv_kernel_size: int = 2,
                 fraud_params: Sequence[FraudLayerParameters] | None = None,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel_size, bias=True)
        self.threshold = 0.0
        if fraud_params is None:
            fraud_params = [FraudLayerParameters(0.0, 0.0, (0.0, 0.0),
                                                  (0.0, 0.0), (0.0, 0.0),
                                                  (0.0, 0.0), (0.0, 0.0),
                                                  (0.0, 0.0))]
        self.fraud = build_fraud_detection_program(fraud_params[0], fraud_params[1:])
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        conv_out = torch.sigmoid(self.conv(data) - self.threshold)
        conv_mean = conv_out.mean(dim=[2, 3])  # flatten spatial dims
        fraud_out = self.fraud(conv_mean)
        ref = torch.zeros_like(fraud_out)
        k_val = self.kernel(fraud_out, ref)
        return k_val
