"""Hybrid fraud detection model that marries classical feature extraction with a quantum expectation head.

The model architecture is a fusion of:
* The two‑layer photonic‑style network from *FraudDetection.py*.
* The hybrid quantum head from *ClassicalQuantumBinaryClassification.py*.
* A fully‑connected quantum layer concept from *FCL.py*.

All classical components are implemented in PyTorch; the quantum circuit is
provided by the companion QML module and injected at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#   Parameter container – extended to carry optional classical weights
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single hybrid layer (photonic + optional linear)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # Optional classical linear weights (used by the fallback FCL style)
    weight: Sequence[Sequence[float]] | None = None
    bias: Sequence[float] | None = None


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _build_linear(params: FraudLayerParameters, clip: bool) -> nn.Module:
    """Construct a 2‑D linear block with optional clipping."""
    weight = torch.tensor(
        params.weight
        if params.weight is not None
        else [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(
        params.bias if params.bias is not None else params.phases,
        dtype=torch.float32,
    )
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


# --------------------------------------------------------------------------- #
#   Differentiable bridge to the quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Forward a tensor through a quantum circuit and back‑propagate via finite‑difference."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        # Expectation values are returned as a 1‑D array
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        eps = 1e-4
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            plus = ctx.circuit.run([val + eps])
            minus = ctx.circuit.run([val - eps])
            grad = (plus - minus) / (2 * eps)
            grad_inputs.append(grad[0])
        grad_tensor = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_tensor * grad_output, None, None


class Hybrid(nn.Module):
    """Quantum expectation head that can be dropped into any PyTorch model."""

    def __init__(self, in_features: int, circuit, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.squeeze(inputs)
        return HybridFunction.apply(inputs, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
#   Main hybrid fraud detector
# --------------------------------------------------------------------------- #
class FraudDetectorHybrid(nn.Module):
    """
    A two‑stage model:
    1. Classical feature extractor built from a stack of photonic‑style layers.
    2. A single‑output quantum expectation head that produces a probability.
    """

    def __init__(self,
                 layer_params: Sequence[FraudLayerParameters],
                 quantum_circuit,
                 shift: float = 0.0) -> None:
        super().__init__()
        if not layer_params:
            raise ValueError("At least one layer parameter set must be supplied.")
        # Build the feature extractor
        layers = list(layer_params)
        self.features = nn.Sequential(
            _build_linear(layers[0], clip=False),
            *(_build_linear(lp, clip=True) for lp in layers[1:])
        )
        # Final linear map to a single logit
        self.classifier = nn.Linear(2, 1)
        # Quantum hybrid head
        self.hybrid = Hybrid(1, quantum_circuit, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        x = self.hybrid(x).unsqueeze(-1)
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["FraudLayerParameters", "FraudDetectorHybrid"]
