"""Quantum fraud detection model using Strawberry Fields variational circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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
    measure: tuple[str,...] = ("photon", "photon")  # Measurement basis


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(modes, params: FraudLayerParameters, *, clip: bool) -> None:
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


class FraudDetectionModel:
    """Quantum fraud detection model that builds a variational SF program and evaluates expectation."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            _apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(q, layer, clip=True)
        return prog

    def evaluate(self, backend: sf.Engine, shots: int = 1000) -> float:
        """Run the program on the specified backend and return a binary classification score."""
        result = backend.run(self.program, shots=shots).meas
        parity = sum([sum(r) % 2 for r in result]) / shots
        return float(parity)

    def train_step(self, optimizer, loss_fn, x, y):
        """Placeholder for a gradient-based training step using SF's autograd."""
        pass


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
