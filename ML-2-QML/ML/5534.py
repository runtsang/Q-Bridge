from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, List, Tuple

# Classical RBF Kernel ----------------------------------------------------
class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0, clip: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.clip = clip

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        out = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        if self.clip:
            out = out.clamp(min=0.0, max=1.0)
        return out

    @staticmethod
    def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        kernel = RBFKernel(gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])

# Fraud‑style linear refinement ------------------------------------------
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

class FraudLinearLayer(nn.Module):
    """Linear + tanh + scaling/shift layer inspired by the fraud‑detection example."""
    def __init__(self,
                 weight: torch.Tensor,
                 bias: torch.Tensor,
                 scale: torch.Tensor,
                 shift: torch.Tensor,
                 clip: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)
        self.clip = clip

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(inputs))
        out = out * self.scale + self.shift
        if self.clip:
            out = out.clamp(-5.0, 5.0)
        return out

    @staticmethod
    def from_params(params: FraudLayerParameters) -> 'FraudLinearLayer':
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
        return FraudLinearLayer(weight, bias, scale, shift, clip=True)

# Classical classifier factory --------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a fully‑connected classifier mirroring the quantum API."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# Unified hybrid classifier -----------------------------------------------
class UnifiedQuantumKernelClassifier(nn.Module):
    """
    Combines a kernel (classical or quantum) with a multi‑layer classifier and an optional fraud‑style refinement.
    """
    def __init__(self,
                 kernel: nn.Module,
                 num_features: int,
                 depth: int = 2,
                 fraud_params: Iterable[FraudLayerParameters] | None = None) -> None:
        super().__init__()
        self.kernel = kernel
        self.classifier, _, _, _ = build_classifier_circuit(num_features, depth)
        self.fraud_layer = None
        if fraud_params is not None:
            # Use only the first layer of parameters for the refinement block
            self.fraud_layer = FraudLinearLayer.from_params(next(iter(fraud_params)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        The input `x` is expected to be a flat feature vector.
        """
        if self.fraud_layer is not None:
            x = self.fraud_layer(x)
        logits = self.classifier(x)
        return logits

    def kernel_matrix(self, a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
        """
        Delegates kernel matrix computation to the underlying kernel implementation.
        """
        if hasattr(self.kernel, 'kernel_matrix'):
            return self.kernel.kernel_matrix(a, b)
        raise AttributeError("Underlying kernel does not expose 'kernel_matrix'.")

__all__ = [
    "RBFKernel",
    "FraudLayerParameters",
    "FraudLinearLayer",
    "build_classifier_circuit",
    "UnifiedQuantumKernelClassifier",
]
