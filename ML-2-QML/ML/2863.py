"""Hybrid fraud detection model combining classical neural network and quantum-inspired RBF kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn

# ----------------------------------------------------------------------
# Data structures
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
    gamma: float = 1.0  # RBF kernel hyper‑parameter

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Classical network construction
# ----------------------------------------------------------------------
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
# RBF kernel module
# ----------------------------------------------------------------------
class RBFKernel(nn.Module):
    """Classical radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# ----------------------------------------------------------------------
# Hybrid model exposing both network and kernel
# ----------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """Hybrid model that can be used with either classical or quantum kernels."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        support_vectors: Sequence[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.network = build_fraud_detection_program(input_params, layers)
        self.kernel = RBFKernel(input_params.gamma)
        self.support_vectors = support_vectors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical network."""
        return self.network(x)

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the RBF kernel matrix between two sets of samples."""
        if y is None:
            y = x
        return self.kernel(x, y)

    def similarity_to_support(self, x: torch.Tensor) -> torch.Tensor:
        """Compute similarities between `x` and a set of support vectors."""
        if self.support_vectors is None:
            raise ValueError("Support vectors not provided")
        return torch.stack([self.kernel_matrix(x, sv) for sv in self.support_vectors], dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "RBFKernel", "FraudDetectionHybrid"]
