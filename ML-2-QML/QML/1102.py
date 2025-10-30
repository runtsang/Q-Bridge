from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, N
from dataclasses import dataclass
from typing import Iterable, Sequence, List

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
    phase_shift: float = 0.0

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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
    # Additional phase‑shift gate that differentiates the quantum model
    for i in range(len(modes)):
        Rgate(params.phase_shift) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

class FraudDetectionModel:
    """
    Quantum photonic fraud‑detection model that mirrors the classical architecture.
    It builds a Strawberry Fields program and offers convenient methods for
    expectation evaluation and parameter conversion.
    """

    def __init__(self, input_params: FraudLayerParameters, layers: List[FraudLayerParameters], dev: str = "fock", cutoff_dim: int = 8) -> None:
        self.input_params = input_params
        self.layers = layers
        self.program = build_fraud_detection_program(input_params, layers)
        self.device = sf.Engine(device=dev, cutoff_dim=cutoff_dim)

    def evaluate(self, inputs: List[tuple[float, float]]) -> List[float]:
        """
        Run the program for each input state (encoded as coherent amplitudes)
        and return the expectation value of the photon number in mode 0.
        """
        expectations = []
        for amp in inputs:
            prog = self.program.copy()
            with prog.context as q:
                Dgate(amp[0], 0) | q[0]
                Dgate(amp[1], 0) | q[1]
            result = self.device.run(prog)
            expectations.append(result.expectation_value(N(0)))
        return expectations

    def to_classical_params(self) -> List[FraudLayerParameters]:
        """
        Strip quantum‑specific phase_shift information and return a list of
        parameters suitable for the classical model.
        """
        return [FraudLayerParameters(
            bs_theta=p.bs_theta,
            bs_phi=p.bs_phi,
            phases=p.phases,
            squeeze_r=p.squeeze_r,
            squeeze_phi=p.squeeze_phi,
            displacement_r=p.displacement_r,
            displacement_phi=p.displacement_phi,
            kerr=p.kerr,
            phase_shift=0.0
        ) for p in [self.input_params] + self.layers]

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionModel"]
