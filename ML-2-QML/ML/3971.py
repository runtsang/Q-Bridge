"""Hybrid kernel and fraud detection module – classical implementation.

This module combines:
- a classical radial‑basis‑function (RBF) kernel,
- a fraud‑detection neural network built from fully‑connected layers,
- and a convenient interface for building and evaluating both.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable
from dataclasses import dataclass

# -------------------- Classical kernel primitives --------------------
class RBFAnsatz(nn.Module):
    """Computes an RBF kernel value between two tensors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class RBFKernel(nn.Module):
    """Wraps :class:`RBFAnsatz` for compatibility with legacy code."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the classical RBF kernel."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# -------------------- Fraud detection primitives --------------------
@dataclass
class FraudLayerParameters:
    """Parameters for a single fully‑connected fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip_value(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out
    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the photonic architecture."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# -------------------- High‑level hybrid class --------------------
class HybridKernelFraudDetector:
    """Unified interface for classical kernels and fraud‑detection models.

    The class exposes:
      * `kernel_matrix(a, b)` – classical RBF Gram matrix.
      * `fraud_model(input_params, layers)` – builds a PyTorch fraud detector.
      * `predict_fraud(model, X)` – runs the detector on a batch of inputs.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    # Kernel utilities
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.gamma)

    # Fraud detection utilities
    def fraud_model(self, input_params: FraudLayerParameters,
                    layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
        return build_fraud_detection_program(input_params, layers)

    @staticmethod
    def predict_fraud(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        """Return raw logits from the fraud‑detection network."""
        with torch.no_grad():
            return model(X)
