"""Quantum fraud detection model using Strawberry Fields.

The module defines a dataclass `FraudLayerParameters` and a class
`FraudDetectionEnhanced` that builds a photonic program with a
variational layer and returns measurement statistics.  The program
supports clipping and a configurable shot count.
"""

from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, List

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

def _apply_layer(modes: List, params: FraudLayerParameters, *, clip: bool) -> None:
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

class FraudDetectionEnhanced:
    """Quantum fraud detection circuit with a variational layer and measurement."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        shots: int = 1024,
    ) -> None:
        self.shots = shots
        self.program = sf.Program(2)
        with self.program.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)

    def run(self, backend: str = "gaussian") -> sf.Result:
        eng = sf.Engine(backend, shots=self.shots)
        return eng.run(self.program)

    def get_measurements(self) -> List[List[float]]:
        res = self.run()
        return res.samples.tolist()

__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
