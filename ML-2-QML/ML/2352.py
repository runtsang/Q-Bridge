"""Hybrid classifier integrating classical feed‑forward layers with photonic fraud‑detection motifs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn

# Re‑use the photonic parameter dataclass from the fraud‑detection seed
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
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class HybridClassifier:
    """
    Classical classifier that mirrors the structure of the quantum and photonic
    fraud‑detection models.  It builds a feed‑forward network whose first
    layer is a fraud‑detection style block followed by a configurable number
    of depth‑wise fully‑connected layers.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers after the fraud layer.
    fraud_params : Iterable[FraudLayerParameters]
        Parameters for the fraud‑detection style block (first layer).
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        fraud_params: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_dim = input_dim
        self.depth = depth
        self.fraud_params = list(fraud_params)

        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        modules: List[nn.Module] = []

        # Fraud‑detection style first layer
        fraud_layer = _layer_from_params(self.fraud_params[0], clip=False)
        modules.append(fraud_layer)

        # Subsequent fully‑connected layers
        in_dim = self.input_dim
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.input_dim)
            modules.append(linear)
            modules.append(nn.ReLU())
            in_dim = self.input_dim

        # Output head
        modules.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_weight_metadata(self) -> Tuple[List[int], List[int]]:
        """Return sizes of each weight matrix and bias vector for introspection."""
        weight_sizes = []
        bias_sizes = []
        for m in self.model:
            if isinstance(m, nn.Linear):
                weight_sizes.append(m.weight.numel())
                bias_sizes.append(m.bias.numel())
        return weight_sizes, bias_sizes

__all__ = ["HybridClassifier", "FraudLayerParameters"]
