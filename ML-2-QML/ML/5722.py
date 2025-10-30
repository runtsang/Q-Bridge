from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np

# ----------------------------------------------------------------------
# Classical building blocks adapted from the original fraud‑detection
# seed.  The layer construction is kept, but the class now accepts a
# clipping flag and exposes a public interface that can be mixed with a
# quantum kernel later on.
# ----------------------------------------------------------------------
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
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Classical RBF kernel utilities – these are now part of the hybrid
# framework and can be swapped with the quantum kernel.
# ----------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Exponentiated quadratic (RBF) ansatz for two‑tensor inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper exposing a callable kernel from :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Hybrid model – a thin wrapper that can use either the classical
# RBF kernel or a quantum kernel (to be provided by the QML module).
# ----------------------------------------------------------------------
class HybridFraudDetector:
    """
    A hybrid fraud‑detection architecture that combines a lightweight
    classical neural network with an optional kernel module.  The
    kernel can be a classical RBF or a quantum kernel supplied by the
    quantum implementation of :class:`HybridFraudDetector`.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        gamma: float = 1.0,
        use_quantum_kernel: bool = False,
    ) -> None:
        self.model = build_fraud_detection_program(input_params, layers)
        self.gamma = gamma
        self.use_quantum_kernel = use_quantum_kernel
        # The quantum kernel will be attached by the QML side via monkey‑patching.
        if not use_quantum_kernel:
            self.kernel: Kernel = Kernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical network."""
        return self.model(x)

    def compute_kernel(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return a Gram matrix using the configured kernel."""
        if self.use_quantum_kernel:
            # The quantum kernel is expected to be injected on the qml side.
            return self.kernel_matrix(a, b)  # type: ignore[no-any-return]
        return kernel_matrix(a, b, self.gamma)

    def predict(self, X: torch.Tensor, training_data: torch.Tensor) -> torch.Tensor:
        """
        Simple prediction routine: compute kernel between X and training data,
        then combine with the neural network output.  For real deployments
        this should be replaced by a proper classifier.
        """
        K = self.compute_kernel(X, training_data)
        out = self.forward(X)
        return out * torch.tensor(K, dtype=out.dtype, device=out.device)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridFraudDetector",
]
