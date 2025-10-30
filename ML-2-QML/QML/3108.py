"""Quantum hybrid estimator combining Qiskit and Strawberry Fields fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
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

class FastHybridEstimator:
    """Hybrid estimator for quantum circuits (Qiskit or Strawberry Fields) with optional shot noise."""

    def __init__(
        self,
        circuit: QuantumCircuit | sf.Program,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if isinstance(circuit, QuantumCircuit):
            self._parameters = list(circuit.parameters)
        else:
            self._parameters = list(circuit.parameters)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | sf.State],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            if isinstance(self.circuit, QuantumCircuit):
                state = Statevector.from_instruction(self._bind_qiskit(params))
                row = [state.expectation_value(obs) for obs in observables]
            else:  # sf.Program
                prog = self._bind_sf(params)
                eng = sf.Engine("gaussian", backend_options={"cutoff_dim": 10})
                result = eng.run(prog)
                state = result.state
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if self.shots is None:
            return results
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                self._rng.normal(0, 1 / self.shots) + val for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    def _bind_qiskit(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _bind_sf(self, parameter_values: Sequence[float]) -> sf.Program:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound program.")
        mapping = dict(zip(self._parameters, parameter_values))
        prog = self.circuit
        prog.assign_parameters(mapping)
        return prog

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FastHybridEstimator"]
