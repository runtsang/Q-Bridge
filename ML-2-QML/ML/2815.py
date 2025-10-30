"""Hybrid fraud detection model combining photonic-inspired layers with
classifier metadata.

The model is a torch.nn.Module that builds a sequential network from a
list of `FraudLayerParameters`.  It exposes the same metadata that the
quantum counterpart produces – an `encoding` list, a `weight_sizes`
list and the `observables` indices – so that downstream training
scripts can treat the classical and quantum versions uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single layer that mimics the photonic circuit."""
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
    *,
    clip: bool = True,
) -> Tuple[nn.Sequential, List[int], List[int]]:
    """Return a sequential network together with metadata.

    Parameters
    ----------
    input_params
        Parameters for the first layer – not clipped.
    layers
        Subsequent layers – clipped to keep the parameters in a realistic range.
    clip
        Whether to clip the parameters of the first layer; default is ``True``.
    """
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    weight_sizes: List[int] = []
    for module in modules:
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    for layer in layers:
        modules.append(_layer_from_params(layer, clip=True))
    for module in modules[1:]:
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    modules.append(nn.Linear(2, 1))
    weight_sizes.append(modules[-1].weight.numel() + modules[-1].bias.numel())
    return nn.Sequential(*modules), list(range(2)), weight_sizes

class FraudDetectionHybrid(nn.Module):
    """Unified classical model that matches the quantum interface.

    The class exposes the same ``encoding`` and ``weight_sizes`` attributes
    as the quantum implementation, allowing a direct comparison of training
    dynamics.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes = build_fraud_detection_program(
            input_params, layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
