"""Hybrid quantum fraud detection model with Strawberry Fields and fast expectation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, Operator

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

class FastBaseEstimator:
    """Evaluate expectation values of operators for a parametrized photonic program."""
    def __init__(self, program: sf.Program, backend: Engine | None = None) -> None:
        self.program = program
        self.backend = backend or Engine("gaussian", backend_options={"seed": 0})
        self.parameters = list(program.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> sf.Program:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound program.")
        param_dict = dict(zip(self.parameters, parameter_values))
        return self.program.bind_parameters(param_dict)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            prog = self._bind(values)
            state = self.backend.run(prog).state
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to photonic expectation values."""
    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [complex(rng.normal(z.real, max(1e-6, 1 / shots)),
                                 rng.normal(z.imag, max(1e-6, 1 / shots))) for z in row]
            noisy.append(noisy_row)
        return noisy

class FraudDetectionHybrid:
    """Hybrid quantum fraud detection model with fast expectation evaluation and optional shot noise."""
    def __init__(self, input_params: FraudLayerParameters, layers: Sequence[FraudLayerParameters]) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.estimator = FastEstimator(self.program)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        return self.estimator.evaluate(
            observables,
            parameter_sets,
            shots=shots,
            seed=seed,
        )
