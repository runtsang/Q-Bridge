from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Union

import numpy as np
import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from strawberryfields.backends import FockBackend
from strawberryfields.state import Statevector

def _clip(value: float, bound: float) -> float:
    """Clamping helper for photonic parameters."""
    return max(-bound, min(bound, value))

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

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    """Insert a full photonic layer into the program."""
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
) -> Program:
    """Create a Strawberry‑Fields program for the fraud‑detection model."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

class FastBaseEstimator:
    """Hybrid estimator that evaluates Strawberry‑Fields programs with optional shot noise."""

    def __init__(self, program: Program | sf.Circuit) -> None:
        self._program = program
        self._backend = FockBackend()
        self._parameters = list(program.parameters)

    def _bind(self, values: Sequence[float]) -> Program:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound program.")
        mapping = dict(zip(self._parameters, values))
        return self._program.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[object],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            prog = self._bind(values)
            if shots is None:
                state = self._backend.run(prog).state
            else:
                state = self._backend.run(prog, shots=shots).state
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noise_row = [rng.normal(0, 1 / shots) + val for val in row]
            noisy.append(noise_row)
        return noisy

__all__ = ["FastBaseEstimator", "FraudLayerParameters", "build_fraud_detection_program"]
