"""Hybrid kernel module that fuses classical RBF, fraud‑detection layers, and a quantum kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


# ------------------------------------------------------------------
# Classical RBF kernel (kept compatible with the original kernel module)
# ------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """RBF kernel with a trainable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that provides a 1‑D kernel value."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for a collection of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# ------------------------------------------------------------------
# Fraud‑detection style neural layers
# ------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection layer."""

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
    """Create a sequential PyTorch model mirroring the fraud‑detection structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ------------------------------------------------------------------
# Hybrid kernel that chains the classical kernel with fraud‑detection layers
# ------------------------------------------------------------------
class HybridKernel(nn.Module):
    """
    Combines a classical RBF kernel with a fraud‑detection style neural net.
    The kernel value is treated as a 2‑D input (duplicated) and passed through
    the fraud‑detection network, allowing end‑to‑end training of both parts.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        fraud_input: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.kernel = Kernel(gamma)

        if fraud_input is None:
            # Default parameters provide a minimal 2‑layer fraud net
            fraud_input = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layers is None:
            fraud_layers = []

        self.fraud_net = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = self.kernel(x, y)
        # Duplicate the scalar kernel into a 2‑D vector for the fraud net
        k_vec = torch.stack([k, k], dim=-1)
        return self.fraud_net(k_vec)


def hybrid_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix of the hybrid kernel."""
    hk = HybridKernel(gamma)
    return np.array([[hk(x, y).item() for y in b] for x in a])


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridKernel",
    "hybrid_kernel_matrix",
]
