"""Hybrid quantum fraud detection program integrating classical encoding and photonic layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

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

class FraudDetectionHybrid:
    """Quantum hybrid fraud detection circuit that encodes classical features,
    applies a sequence of FraudLayerParameters-based photonic layers,
    and measures all modes in the photon-number basis.
    """
    def __init__(self, layer_params: Iterable[FraudLayerParameters] | None = None):
        self.layer_params = layer_params or []

    def encode(self, program: sf.Program, features: Sequence[float]) -> None:
        """Encode classical features into the photonic modes using Rgate."""
        with program.context as q:
            for i, f in enumerate(features):
                Rgate(f) | q[i]

    def forward(self, features: Sequence[float]) -> sf.State:
        """Build and run the circuit for a single feature vector."""
        prog = sf.Program(2)
        self.encode(prog, features)
        with prog.context as q:
            for layer in self.layer_params:
                _apply_layer(q, layer, clip=True)
        eng = sf.Engine("gaussian", backend_options={"seed": 42})
        result = eng.run(prog, shots=1)
        return result.state

    def get_features(self, state: sf.State) -> list[float]:
        """Return expectation values of photon number for each mode."""
        return [state.expectation_value("n", mode=i) for i in range(state.n_wires)]

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
