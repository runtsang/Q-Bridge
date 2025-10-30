"""Variational photonic fraud‑detection layer using StrawberryFields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import strawberryfields as sf
from strawberryfields.ops import (
    BSgate,
    Dgate,
    Kgate,
    Rgate,
    Sgate,
    N,
)


@dataclass
class FraudLayerParameters:
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


def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    hidden_params: Iterable[FraudLayerParameters],
    *,
    clip_weights: bool = True,
) -> sf.Program:
    """Create a StrawberryFields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in hidden_params:
            _apply_layer(q, layer, clip=clip_weights)
    return program


class FraudDetectionQuantumLayer(torch.nn.Module):
    """Variational photonic layer implemented with StrawberryFields, wrapped as an nn.Module."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        *,
        clip_weights: bool = True,
        cutoff_dim: int = 10,
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.hidden_params = list(hidden_params)
        self.clip_weights = clip_weights
        self.cutoff_dim = cutoff_dim
        self.backend = sf.SFBackend("fock", cutoff_dim=cutoff_dim)

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            _apply_layer(q, self.input_params, clip=False)
            for params in self.hidden_params:
                _apply_layer(q, params, clip=self.clip_weights)
        return prog

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute photon‑number expectation values for a single input sample."""
        prog = self._build_program()
        # Encode the two‑dimensional input into displacements on the two modes
        for i, val in enumerate(x):
            Dgate(val.item(), 0) | prog[0][i]
        self.backend.reset()
        self.backend.run(prog)
        exp_val = self.backend.expectation_value(N(0)) + self.backend.expectation_value(N(1))
        return torch.tensor(exp_val, dtype=x.dtype)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionQuantumLayer",
]
