from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import torch
from torch import nn
import numpy as np

# --- Photonic‑style layer parameters (kept for compatibility) ---
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

def _layer_from_params(params: FraudLayerParameters, clip: bool = True) -> nn.Module:
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
            return out * self.scale + self.shift

    return Layer()

# --- QCNN feature extractor (classical analogue of the quantum block) ---
class QCNNModel(nn.Module):
    """Feature extractor inspired by quantum convolution layers."""
    def __init__(self, depth: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, hidden_dim), nn.Tanh())
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh()))
            self.blocks.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh()))
            hidden_dim //= 2
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))

# --- Classical RBF kernel (compatible with the quantum interface) ---
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --- Hybrid factory that stitches everything together ---
class FraudDetectionHybrid:
    """
    Combines photonic‑style layers, a QCNN feature extractor, and an RBF kernel
    into a single classical pipeline.  Mirrors the original FraudDetection API
    while exposing additional quantum‑inspired modules.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 qcnn_depth: int = 3,
                 kernel_gamma: float = 1.0) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.qcnn_depth = qcnn_depth
        self.kernel_gamma = kernel_gamma

    def build_fraud_model(self) -> nn.Sequential:
        """Return a sequential network consisting of the fraud layers followed by a QCNN."""
        modules = [_layer_from_params(self.input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in self.layers)
        modules.append(nn.Linear(2, 1))
        modules.append(QCNNModel(depth=self.qcnn_depth))
        return nn.Sequential(*modules)

    def compute_kernel(self,
                       a: Sequence[torch.Tensor],
                       b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix using the configured RBF kernel."""
        return kernel_matrix(a, b, gamma=self.kernel_gamma)

__all__ = ["FraudLayerParameters", "QCNNModel", "RBFKernel",
           "FraudDetectionHybrid"]
