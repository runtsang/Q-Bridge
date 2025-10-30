"""Hybrid estimator combining classical fraud‑detection layers with optional quantum feature extraction.

This module defines `HybridEstimatorQNN`, a torch.nn.Module that
- builds a classical feed‑forward network mirroring the photonic
  fraud‑detection architecture;
- optionally attaches a quantum circuit (via a user supplied Estimator)
  whose parameters are treated as additional trainable weights;
- enforces weight clipping as in the photonic example to keep
  parameters within a physically realistic range.

The scaling paradigm is a *combination* of layer‑wise clipping
and end‑to‑end quantum‑classical training.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Callable, Optional

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32
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

class HybridEstimatorQNN(nn.Module):
    """
    A hybrid neural network that combines a classical fraud‑detection
    style feed‑forward architecture with optional quantum feature
    extraction.  The quantum part is represented by an Estimator
    callable that must return a tensor of shape (batch, 1).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        quantum_estimator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.classical_net = build_fraud_detection_program(input_params, layers)
        self.quantum_estimator = quantum_estimator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sum of classical output and quantum expectation."""
        classical = self.classical_net(x)
        if self.quantum_estimator is not None:
            quantum = self.quantum_estimator(x)
            return classical + quantum
        return classical

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "HybridEstimatorQNN"]
