from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Photonic‑inspired layer parameters."""
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

def _apply_convolution(modes, kernel_size: int = 2, threshold: float = 0.5) -> None:
    """Photonic convolutional block that mixes neighbouring modes."""
    # Simple 2‑mode interferometer network
    for i in range(0, len(modes)-1, 2):
        BSgate(0.5, 0.0) | (modes[i], modes[i+1])
    # Conditional displacement mimicking a filter response
    for i, mode in enumerate(modes):
        Dgate(0.2 * threshold, 0.0) | mode

def _apply_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
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

def build_fraud_detection_quantum_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Creates a Strawberry Fields program starting with a photonic convolution
    block, followed by photonic layers mirroring the classical counterpart."""
    program = sf.Program(2)
    with program.context as q:
        _apply_convolution(q, kernel_size=2, threshold=0.5)
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

class FraudDetectionHybrid:
    """Quantum‑only hybrid model: convolution‑like photonic block + photonic layers."""
    def __init__(self, input_params: FraudLayerParameters,
                 layers: list[FraudLayerParameters]) -> None:
        self.program = build_fraud_detection_quantum_program(input_params, layers)

    def run(self, shots: int = 1024) -> sf.result.Result:
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        return eng.run(self.program, shots=shots)

__all__ = ["FraudLayerParameters", "build_fraud_detection_quantum_program", "FraudDetectionHybrid"]
