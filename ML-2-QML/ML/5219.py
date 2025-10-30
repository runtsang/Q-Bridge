import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

class KernalAnsatz(nn.Module):
    """Classical radial basis function ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def SamplerQNN():
    """A lightweight softmax sampler built from a small NN."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(inputs), dim=-1)
    return SamplerModule()

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

class HybridKernelMethod(nn.Module):
    """
    Hybrid kernel method that combines a classical RBF kernel, an optional softmax sampler,
    and optional fraud‑detection style layers.  The class mirrors the API of the original
    QuantumKernelMethod but is fully classical.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        use_sampler: bool = False,
        fraud_params: Tuple[FraudLayerParameters,...] | None = None,
    ) -> None:
        super().__init__()
        self.kernel = Kernel(gamma)
        self.sampler = SamplerQNN() if use_sampler else None
        self.fraud_model = (
            build_fraud_detection_program(fraud_params[0], fraud_params[1:])
            if fraud_params
            else None
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value between two feature vectors."""
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two collections of feature vectors."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce a two‑class probability vector using the optional sampler or fraud model."""
        if self.sampler is not None:
            return self.sampler(inputs)
        if self.fraud_model is not None:
            return self.fraud_model(inputs)
        return torch.sigmoid(inputs)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "SamplerQNN",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridKernelMethod",
]
