import math
from dataclasses import dataclass
from typing import Iterable, List

import strawberryfields as sf
from strawberryfields.ops import (
    BSgate,
    Dgate,
    Rgate,
    Sgate,
    Kgate,
    MeasureFock,
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
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


class FraudDetectionQuantumCircuit:
    """
    A Strawberry Fields program that implements the photonic fraud‑detection
    circuit described in the seed.  Each layer applies a symmetric beam splitter,
    single‑mode rotations, squeezers, displacements and Kerr non‑linearity.
    """
    def __init__(self, layers: List[FraudLayerParameters], clip: bool = True) -> None:
        self.layers = layers
        self.clip = clip

    def build(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            for layer in self.layers:
                self._apply_layer(q, layer)
        return prog

    def _apply_layer(self, modes: List[sf.ops.Qubit], params: FraudLayerParameters) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]

        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not self.clip else _clip(r, 5), phi) | modes[i]

        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]

        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not self.clip else _clip(r, 5), phi) | modes[i]

        for i, k in enumerate(params.kerr):
            Kgate(k if not self.clip else _clip(k, 1)) | modes[i]

        # Typical measurement: Fock count on the first mode
        MeasureFock() | modes[0]


__all__ = ["FraudLayerParameters", "FraudDetectionQuantumCircuit"]
