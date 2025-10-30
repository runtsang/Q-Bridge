"""Classical fraud detection model with residual connections, dropout, and layer normalization.

The model mirrors the photonic architecture but adds classical regularization
techniques to improve generalisation.  The shared class name `FraudDetectionHybrid`
enables direct comparison with the quantum counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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

def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout: float = 0.0,
    use_layernorm: bool = False,
) -> nn.Module:
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
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.layernorm = nn.LayerNorm(2) if use_layernorm else nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = x * self.scale + self.shift
            x = self.dropout(x)
            x = self.layernorm(x)
            return x

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout: float = 0.0,
    use_layernorm: bool = False,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure with optional regularisation."""
    modules = [
        _layer_from_params(input_params, clip=False, dropout=dropout, use_layernorm=use_layernorm)
    ]
    modules.extend(
        _layer_from_params(layer, clip=True, dropout=dropout, use_layernorm=use_layernorm)
        for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionHybrid(nn.Module):
    """Classical fraud detection model with optional regularisation.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    dropout : float, optional
        Dropout probability applied after each layer.
    use_layernorm : bool, optional
        Whether to apply LayerNorm after each layer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(
            input_params,
            layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(inputs)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
