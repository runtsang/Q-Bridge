from dataclasses import dataclass
import torch
from torch import nn
import numpy as np

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

def _build_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_build_layer(input_params, clip=False)]
    modules.extend(_build_layer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection architecture that combines a classical feed‑forward network
    (photonic inspired) with a kernel module (classical or quantum)."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        support_vectors: torch.Tensor,
        kernel: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.classical_net = build_fraud_detection_program(input_params, layers)
        self.register_buffer("support_vectors", support_vectors)
        self.kernel = kernel or Kernel()
        self.classifier = nn.Linear(2 + support_vectors.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_feat = self.classical_net(x).flatten(-1)  # shape (batch, 2)
        kernel_feat = self.kernel(x, self.support_vectors)  # shape (batch, n_support)
        combined = torch.cat([class_feat, kernel_feat], dim=-1)
        return self.classifier(combined)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "KernalAnsatz", "Kernel", "kernel_matrix", "FraudDetectionHybrid"]
