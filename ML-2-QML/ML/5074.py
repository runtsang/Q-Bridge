"""
Classical hybrid fraud‑detection module integrating attention and quantum kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Parameters describing a photonic‑like layer
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# 2. Classical attention helper (from SelfAttention.py)
# --------------------------------------------------------------------------- #
def SelfAttention(embed_dim: int):
    class ClassicalSelfAttention:
        def __init__(self) -> None:
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            query = torch.as_tensor(
                inputs @ rotation_params.reshape(self.embed_dim, -1),
                dtype=torch.float32,
            )
            key = torch.as_tensor(
                inputs @ entangle_params.reshape(self.embed_dim, -1),
                dtype=torch.float32,
            )
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention()

# --------------------------------------------------------------------------- #
# 3. Classical RBF kernel (from QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 4. Helper to build a classical layer mirroring the photonic circuit
# --------------------------------------------------------------------------- #
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# --------------------------------------------------------------------------- #
# 5. Hybrid fraud‑detection model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    Classical hybrid fraud‑detection model.

    Parameters
    ----------
    input_params
        Parameters of the first photonic layer (no clipping).
    layers
        Iterable of subsequent layer parameters (clipped).
    attention_params
        Dictionary with ``rotation_params`` and ``entangle_params`` for the
        self‑attention block.
    gamma
        RBF kernel bandwidth.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        attention_params: dict[str, np.ndarray],
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(embed_dim=attention_params["embed_dim"])(attention_params["embed_dim"])
        self.attention_params = attention_params
        self.layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)] +
            [_layer_from_params(l, clip=True) for l in layers]
        )
        self.kernel = Kernel(gamma)
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inputs through self‑attention, photonic layers and a linear head."""
        # Attention block
        att_out = self.attention.run(
            self.attention_params["rotation_params"],
            self.attention_params["entangle_params"],
            x.numpy(),
        )
        x = torch.from_numpy(att_out).float()

        # Photonic layers
        for layer in self.layers:
            x = layer(x)

        # Final classifier
        return self.classifier(x)

    def kernel_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return RBF kernel similarity between two tensors."""
        return self.kernel(x, y)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybrid",
    "kernel_matrix",
]
