"""Hybrid fraud detection model combining photonic-inspired classical layers
and a fully‑connected quantum sub‑layer.

The architecture mirrors the original photonic construction but inserts a
parameterised quantum circuit after every classical layer.  The quantum
circuit is simulated classically so that the module remains pure PyTorch
and can be used without a quantum backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import math

@dataclass
class PhotonicLayerParameters:
    """Parameters describing a single photonic layer."""
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

def _layer_from_params(params: PhotonicLayerParameters, *, clip: bool) -> nn.Module:
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

class FCLQuantumLayer(nn.Module):
    """Classical simulation of a fully‑connected quantum layer.

    Implements the expectation value of a single‑qubit Ry(θ) circuit
    followed by a Z measurement.  For a list of angles the output is
    the mean expectation over all qubits.
    """
    def __init__(self, n_qubits: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape (batch, n_qubits)
        thetas = thetas.view(-1, self.n_qubits)
        cos_vals = torch.cos(thetas)
        expectation = cos_vals.mean(dim=1, keepdim=True)
        return expectation

class FraudDetectionHybridModel(nn.Module):
    """Hybrid fraud‑detection architecture.

    The model alternates a photonic‑inspired classical layer with a
    fully‑connected quantum sub‑layer, culminating in a single‑output
    linear head.
    """
    def __init__(
        self,
        input_params: PhotonicLayerParameters,
        layers: Iterable[PhotonicLayerParameters],
        n_qubits: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # initial photonic block
        self.layers.append(_layer_from_params(input_params, clip=False))
        self.layers.append(FCLQuantumLayer(n_qubits))
        # subsequent blocks
        for layer in layers:
            self.layers.append(_layer_from_params(layer, clip=True))
            self.layers.append(FCLQuantumLayer(n_qubits))
        self.final_linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for module in self.layers:
            out = module(out)
        out = self.final_linear(out)
        return out

__all__ = [
    "PhotonicLayerParameters",
    "FCLQuantumLayer",
    "FraudDetectionHybridModel",
]
