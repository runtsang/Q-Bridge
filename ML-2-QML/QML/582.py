"""Strawberry Fields program used in the fraud detection example, extended to return a probability vector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureFock


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
    """Clip values to keep the photonic parameters within a safe range."""
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
    """Create a Strawberry Fields program that ends with a Fock measurement on mode 0."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
        # Measurement: Fock measurement on mode 0
        MeasureFock() | q[0]
    return program


def sample_fraud_detection(program: sf.Program, shots: int = 1024) -> np.ndarray:
    """Run the program on a Fock simulator and return a probability vector over 0/1 photon counts in mode 0."""
    import numpy as np
    from strawberryfields import Simulator
    sim = Simulator("fock", dim=2, shots=shots)
    results = sim.run(program).shots
    counts = {0: 0, 1: 0}
    for n0, _ in results:
        if n0 in counts:
            counts[n0] += 1
    probs = np.array([counts[0] / shots, counts[1] / shots], dtype=np.float32)
    return probs


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "sample_fraud_detection"]
