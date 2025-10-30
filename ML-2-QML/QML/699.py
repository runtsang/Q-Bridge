"""Enhanced photonic fraud detection circuit with variational entanglement and measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    # Optional pre-rotation to enrich feature mapping
    Rgate(params.bs_phi) | modes[0]
    Rgate(params.bs_phi) | modes[1]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]

    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]

    # Additional entangling gate with doubled phase
    BSgate(params.bs_theta, 2 * params.bs_phi) | (modes[0], modes[1])

    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]

    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]

    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

    # Measurement in the Fock basis for each mode
    sf.measure_fock(modes[0])
    sf.measure_fock(modes[1])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model with variational layers."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


__all__ = ["build_fraud_detection_program"]
