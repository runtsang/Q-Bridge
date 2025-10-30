"""Unified hybrid layer combining classical fully‑connected and photonic fraud‑detection motifs."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable

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

class UnifiedHybridLayer(nn.Module):
    """
    Classical‑only hybrid layer that mimics the structure of a quantum‑style
    fully‑connected layer and a photonic fraud‑detection network.
    """

    def __init__(
        self,
        classical_params: FraudLayerParameters,
        n_qubits: int = 1,
        clip: bool = False,
    ) -> None:
        super().__init__()
        self.classical_submodule = _layer_from_params(classical_params, clip=clip)
        self.n_qubits = n_qubits

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the classical output and return a dummy quantum expectation
        (all zeros) to keep the interface compatible with the quantum version.
        """
        _ = torch.tensor(list(thetas), dtype=torch.float32)
        quantum_expectation = np.zeros(self.n_qubits)
        dummy_input = torch.zeros(2, dtype=torch.float32)
        classical_output = self.classical_submodule(dummy_input).detach().numpy()
        return np.concatenate([classical_output, quantum_expectation], axis=0)

__all__ = ["UnifiedHybridLayer", "FraudLayerParameters"]
