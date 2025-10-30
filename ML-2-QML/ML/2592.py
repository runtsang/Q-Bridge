"""Hybrid fraud detection model combining classical neural network with quantum kernel.

The module preserves the original `FraudLayerParameters` and
`build_fraud_detection_program` API for compatibility, while adding a new
`FraudDetectionHybrid` class that merges a classical feed‑forward network
with a kernel layer.  The kernel can be a standard RBF kernel or a quantum
kernel implemented in the accompanying QML module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Callable

import torch
from torch import nn

# ----------------------------------------------------------------------
# Original classes retained for backward compatibility
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# New hybrid model
# ----------------------------------------------------------------------
# Optional import of the quantum kernel – falls back to a dummy if unavailable.
try:
    from.FraudDetection__gen150_qml import QuantumFraudKernel
except Exception:  # pragma: no cover
    QuantumFraudKernel = None

class KernelLayer(nn.Module):
    """Kernel layer that evaluates pairwise similarities using a callable kernel."""
    def __init__(self, kernel_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.kernel_func = kernel_func
        self.register_buffer("support", torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store support vectors if not yet stored
        if self.support.size(0) == 0:
            self.support = x.detach()
        # Compute Gram matrix between x and support
        gram = torch.zeros(x.size(0), self.support.size(0), device=x.device)
        for i, xi in enumerate(x):
            for j, sj in enumerate(self.support):
                gram[i, j] = self.kernel_func(xi, sj)
        return gram

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model using a classical network + kernel layer."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        quantum_kernel: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Tanh()

        # Decide on kernel function
        if kernel_type == "rbf":
            kernel_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
                lambda x, y: torch.exp(-gamma * torch.sum((x - y) ** 2))
            )
        elif kernel_type == "quantum" and quantum_kernel and QuantumFraudKernel is not None:
            self.q_kernel = QuantumFraudKernel()
            kernel_func = lambda x, y: self.q_kernel.kernel(x.unsqueeze(0), y.unsqueeze(0))
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

        self.kernel_layer = KernelLayer(kernel_func)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        k = self.kernel_layer(h)
        out = self.classifier(k)
        return out

    def compute_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix between X and Y using the selected kernel."""
        kernel_func = self.kernel_layer.kernel_func
        return torch.stack([torch.stack([kernel_func(xi, yj) for yj in Y]) for xi in X])

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
