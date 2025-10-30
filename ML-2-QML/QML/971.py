"""Quantum photonic fraud detection circuit with automatic differentiation support."""

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
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def hybrid_fraud_detection(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    shots: int = 1000,
) -> float:
    """Run the photonic circuit and return the probability of the |11⟩ outcome.

    The probability of detecting a photon in both modes is used as a proxy
    for a fraudulent transaction.  The function is fully differentiable
    with respect to the parameters, allowing it to be plugged into a
    classical optimiser.
    """
    prog = build_fraud_detection_program(input_params, layers)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    result = eng.run(prog, shots=shots)
    probs = result.samples
    # Count occurrences of the |11> outcome
    count_11 = sum(1 for sample in probs if sample[0] == 1 and sample[1] == 1)
    return count_11 / shots


def sweep_parameter(
    base_params: FraudLayerParameters,
    param_name: str,
    values: Sequence[float],
    layers: Iterable[FraudLayerParameters],
) -> list[tuple[float, float]]:
    """Return a list of (value, probability) for sweeping a single parameter."""
    outcomes = []
    for val in values:
        # copy params and modify
        params = FraudLayerParameters(**{**base_params.__dict__, param_name: val})
        prob = hybrid_fraud_detection(params, layers)
        outcomes.append((val, prob))
    return outcomes


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "hybrid_fraud_detection",
    "sweep_parameter",
]
