"""Hybrid quantum fraud detection model with fast batched estimation and shot noise.

The quantum program mirrors the photonic architecture from the original seed
and exposes a unified FastEstimator that can inject Gaussian shot noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields import ops


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


def _apply_layer(
    modes: Sequence[ops.Mode],
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        ops.Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        ops.Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        ops.Kgate(k if not clip else _clip(k, 1)) | modes[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog


class FastBaseEstimator:
    """Deterministic batched evaluator for a Strawberry Fields program."""
    def __init__(self, program: sf.Program, engine: Engine) -> None:
        self.program = program
        self.engine = engine

    def evaluate(
        self,
        observables: Iterable[Callable[[sf.Result], complex]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_prog = self.program.assign_parameters(
                dict(zip(self.program.parameters, params)), inplace=False
            )
            result = self.engine.run(bound_prog)
            row = [obs(result) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[sf.Result], complex]],
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
            noisy_row = [rng.normal(float(val), max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


class FraudDetectionModel:
    """Quantum fraud‑detection circuit with a fast evaluation interface."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        cutoff_dim: int = 8,
    ) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.engine = Engine("fock", backend_options={"cutoff_dim": cutoff_dim})

    def evaluate(
        self,
        observables: Iterable[Callable[[sf.Result], complex]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate the circuit on a batch of parameter sets, optionally adding noise."""
        estimator = FastEstimator(self.program, self.engine, shots=shots, seed=seed)
        return estimator.evaluate(observables, parameter_sets)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FastBaseEstimator",
    "FastEstimator",
    "FraudDetectionModel",
]
