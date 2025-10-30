"""Quantum fraud detection using Strawberry Fields.

The class `FraudDetectionHybrid` constructs a photonic variational circuit
with multiple layers of beam‑splitter, squeezing, displacement, and Kerr
gates.  Each layer can clip parameters to keep the simulation stable.
The circuit is evaluated with a fast state‑vector estimator that supports
shot‑noise simulation.
"""

from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List

# --------------------------------------------------------------------------- #
#  Parameter container
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
#  Fast estimator
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluates expectation values of observables on a Strawberry Fields program."""
    def __init__(self, program: sf.Program, cutoff: int = 12) -> None:
        self.program = program
        self.engine = Engine("gaussian", cutoff=cutoff)

    def evaluate(self, observables: Iterable[Callable[[sf.State], float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            bound_prog = self.program.bind_parameters(dict(zip(self.program.parameters, params)))
            state = self.engine.run(bound_prog).state
            row = [obs(state) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#  FraudDetectionHybrid quantum model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid:
    """Photonic variational circuit for fraud detection."""
    def __init__(self, n_modes: int = 2, cutoff: int = 12) -> None:
        self.n_modes = n_modes
        self.cutoff = cutoff
        self.engine = Engine("gaussian", cutoff=self.cutoff)
        self.program = sf.Program(self.n_modes)

    def _apply_layer(self, q, params: FraudLayerParameters, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | q[i]

    def build_program(self, input_params: FraudLayerParameters,
                      layers: Iterable[FraudLayerParameters]) -> sf.Program:
        with self.program.context as q:
            self._apply_layer(q, input_params, clip=False)
            for layer in layers:
                self._apply_layer(q, layer, clip=True)
        return self.program

    def evaluate(self, parameter_sets: Sequence[Sequence[float]],
                 observables: Iterable[Callable[[sf.State], float]]) -> List[List[float]]:
        estimator = FastBaseEstimator(self.program, cutoff=self.cutoff)
        return estimator.evaluate(observables, parameter_sets)

# --------------------------------------------------------------------------- #
#  Compatibility helper
# --------------------------------------------------------------------------- #

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> sf.Program:
    """Compatibility wrapper that returns a Strawberry Fields program."""
    hybrid = FraudDetectionHybrid()
    return hybrid.build_program(input_params, layers)

__all__ = ["FraudDetectionHybrid", "FraudLayerParameters",
           "build_fraud_detection_program", "FastBaseEstimator"]
