"""Hybrid fraud detection model combining classical neural layers with a quantum evaluation layer.

The model mirrors the photonic fraud detection architecture from the original seed, but replaces the
photonic layer with a Qiskit‑based quantum circuit.  Classical layers are built with
parameter clipping to emulate experimental bounds, and the quantum evaluator can be swapped
at runtime for a different backend or a classical fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, Optional

import torch
from torch import nn
import numpy as np


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
    """Clip a scalar value to the specified bound."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single 2‑input → 2‑output linear layer with optional clipping."""
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


class ClassicalFallback(nn.Module):
    """A lightweight classical replacement for the quantum layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the first feature as the input to the linear map
        values = x[:, 0].unsqueeze(1)
        return torch.tanh(self.linear(values))


class QuantumLayerWrapper(nn.Module):
    """Wrap a callable quantum evaluator so it can be used as an nn.Module."""
    def __init__(self, evaluator: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__()
        self.evaluator = evaluator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            np_input = x.detach().cpu().numpy()
            np_output = self.evaluator.evaluate(np_input)
            return torch.from_numpy(np_output).to(x.device)


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (unclipped) classical layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent clipped layers.
    quantum_evaluator : Callable[[np.ndarray], np.ndarray], optional
        Quantum evaluator that accepts a 2‑D numpy array and returns a 2‑D array
        of quantum expectations.  If omitted, a classical fallback is used.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        quantum_evaluator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.classical_layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)] +
            [_layer_from_params(l, clip=True) for l in layers]
        )
        self.output_layer = nn.Linear(2, 1)

        if quantum_evaluator is None:
            fallback = ClassicalFallback(n_features=1)
            self.quantum_layer = QuantumLayerWrapper(fallback)
        else:
            self.quantum_layer = QuantumLayerWrapper(quantum_evaluator)

    def set_quantum_evaluator(
        self, evaluator: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """Replace the underlying quantum evaluator at runtime."""
        self.quantum_layer.evaluator = evaluator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.classical_layers:
            x = layer(x)
        quantum_out = self.quantum_layer(x)
        # Augment classical features with quantum output
        combined = x + quantum_out
        return self.output_layer(combined)


__all__ = ["FraudDetectionHybrid", "FraudLayerParameters", "QuantumLayerWrapper", "ClassicalFallback"]
