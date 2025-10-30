"""Unified kernel‑based fraud detection – classical implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Classical RBF kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Learnable radial basis function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes a single‑value kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute a Gram matrix from two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Fraud‑detection network
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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Unified hybrid model
# --------------------------------------------------------------------------- #
class UnifiedKernelFraudDetector(nn.Module):
    """Combines a kernel (classical or quantum) with a fraud‑detection head."""

    def __init__(
        self,
        *,
        gamma: float | None = None,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
        kernel_type: str = "classical",
    ) -> None:
        super().__init__()
        if kernel_type == "classical":
            self.kernel = Kernel(gamma if gamma is not None else 1.0)
        else:
            raise ValueError("Only classical kernel is available in the ML module.")
        if fraud_params is None:
            raise ValueError("fraud_params must be provided.")
        params = list(fraud_params)
        self.fraud_net = build_fraud_detection_program(params[0], params[1:])

    def forward(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute Gram matrix and classify it."""
        gram = kernel_matrix(a, b, gamma=self.kernel.ansatz.gamma.item())
        # Flatten Gram matrix into a feature vector
        features = torch.from_numpy(gram).float()
        # The fraud net expects a batch of 2‑dim inputs; we reshape accordingly
        dummy = torch.zeros_like(features)
        inputs = torch.stack([features, dummy], dim=-1).view(-1, 2)
        return self.fraud_net(inputs)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "UnifiedKernelFraudDetector",
]
