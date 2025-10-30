"""Strawberry Fields program with a rotation‑gate variational layer and Z‑operator readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields.ops import Rgate, Dgate, Kgate, Sgate, BSgate, MeasureZ


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
    rotation_angles: tuple[float, float]  # new rotation parameters for the variational circuit


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam splitter
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

    # Phase shifts
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]

    # Squeezing
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]

    # Second beam splitter
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

    # Phase shifts again
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]

    # Displacement
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]

    # Kerr nonlinearity
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

    # Variational rotation layer (new)
    for i, angle in enumerate(params.rotation_angles):
        Rgate(angle) | modes[i]

    # Measurement readout
    MeasureZ() | modes[0]
    MeasureZ() | modes[1]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


__all__ = ["build_fraud_detection_program", "FraudLayerParameters"]
