"""
Hybrid classical kernel combining RBF with fraud-detection inspired neural layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
import torch.nn as nn
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters for a fraud‑detection style fully connected layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Build a single neural layer from the supplied parameters."""
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
    """Create a sequential PyTorch model mirroring the fraud‑detection layers."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudQuantumKernel(nn.Module):
    """
    Classical kernel that augments an RBF kernel with a fraud‑detection neural network.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        fraud_params: FraudLayerParameters | None = None,
        layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.fraud_net: nn.Module | None = None
        if fraud_params is not None:
            self.fraud_net = build_fraud_detection_program(fraud_params, layers or [])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute the hybrid kernel value between two feature vectors.
        """
        # Classical RBF component
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

        if self.fraud_net is None:
            return rbf.squeeze()

        # Fraud‑detection network component
        fraud_x = self.fraud_net(x).squeeze()
        fraud_y = self.fraud_net(y).squeeze()
        return (rbf * fraud_x * fraud_y).squeeze()

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the Gram matrix for two collections of vectors."""
        return np.array(
            [[self.forward(x, y).item() for y in b] for x in a]
        )


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudQuantumKernel"]
