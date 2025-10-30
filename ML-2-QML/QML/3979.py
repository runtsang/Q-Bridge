"""Quantum‑centric implementation of the fraud detection hybrid model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Re‑declare the parameter dataclass to keep the module self‑contained
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


class FraudDetectionHybrid:
    """Augmented hybrid model that also builds a StrawberryFields program and evaluates quantum observables."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.qprog = self._build_qprog(input_params, layers)

    def _build_qprog(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> sf.Program:
        program = sf.Program(2)
        with program.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return program

    def evaluate_quantum(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute quantum expectation values for each parameter set."""
        # Build a simple QuantumCircuit wrapper for demonstration
        qc = QuantumCircuit(2)
        # Placeholder: actual conversion from sf.Program to Qiskit circuit is non‑trivial
        # and omitted for brevity. In practice, use a dedicated transpiler.

        class FastQuantumEstimator:
            def __init__(self, circuit: QuantumCircuit):
                self._circuit = circuit
                self._parameters = list(circuit.parameters)

            def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
                mapping = dict(zip(self._parameters, parameter_values))
                return self._circuit.assign_parameters(mapping, inplace=False)

            def evaluate(self, obs: Iterable[BaseOperator], params: Sequence[Sequence[float]]) -> List[List[complex]]:
                results: List[List[complex]] = []
                for values in params:
                    state = Statevector.from_instruction(self._bind(values))
                    row = [state.expectation_value(o) for o in obs]
                    results.append(row)
                return results

        estimator = FastQuantumEstimator(qc)
        return estimator.evaluate(observables, parameter_sets)
