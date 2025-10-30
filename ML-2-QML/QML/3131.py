"""Quantum photonic fraud detection model using Strawberry Fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑inspired fully‑connected layer."""
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

def _apply_layer(q: list, params: FraudLayerParameters, clip: bool = False) -> None:
    # Beam splitter
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    # Phase rotations
    Rgate(params.phases[0]) | q[0]
    Rgate(params.phases[1]) | q[1]
    # Squeezing
    Sgate(_clip(params.squeeze_r[0], 5), params.squeeze_phi[0]) | q[0]
    Sgate(_clip(params.squeeze_r[1], 5), params.squeeze_phi[1]) | q[1]
    # Second beam splitter
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    # Phase rotations again
    Rgate(params.phases[0]) | q[0]
    Rgate(params.phases[1]) | q[1]
    # Displacement
    Dgate(_clip(params.displacement_r[0], 5), params.displacement_phi[0]) | q[0]
    Dgate(_clip(params.displacement_r[1], 5), params.displacement_phi[1]) | q[1]
    # Kerr
    Kgate(_clip(params.kerr[0], 1)) | q[0]
    Kgate(_clip(params.kerr[1], 1)) | q[1]

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

class FraudDetectionQuantumModel:
    """Wrapper for running the photonic fraud detection program on a simulator."""
    def __init__(self, simulator: sf.Simulator = None):
        self.sim = simulator or sf.Simulator('gaussian')

    def forward(self,
                input_params: FraudLayerParameters,
                layers: Iterable[FraudLayerParameters],
                shots: int = 1024) -> float:
        prog = build_fraud_detection_program(input_params, layers)
        results = self.sim.run(prog, shots=shots)
        # Example observable: photon number in mode 0
        return results.state.expectation_value(sf.ops.Number(0))

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionQuantumModel"]
