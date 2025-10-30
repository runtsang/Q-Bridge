"""Minimal estimator primitive used by the simplified fast primitives package.

Extended to support backend selection, shot‑noise simulation, and
automatic parameter‑shift gradients for hybrid quantum‑classical workflows."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.result import Result


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(self, circuit: QuantumCircuit, backend_name: str = "statevector_simulator") -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend_name
        self.backend = AerSimulator(method="statevector") if backend_name == "statevector_simulator" else AerSimulator()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[Operator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_shots(self, observables: Iterable[Operator], parameter_sets: Sequence[Sequence[float]], shots: int = 1024) -> List[List[complex]]:
        """Compute shot‑noised expectation values using the AerSimulator."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            # Attach expectation-value measurements for each observable
            for obs in observables:
                bound.save_expectation_value(obs, shots=shots)
            job = self.backend.run(bound, shots=shots)
            result: Result = job.result()
            row = [result.get_expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _expectation_value(self, observable: Operator, parameter_values: Sequence[float]) -> complex:
        bound = self._bind(parameter_values)
        state = Statevector.from_instruction(bound)
        return state.expectation_value(observable)

    def gradient(self, observable: Operator, parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Return the gradient of an observable w.r.t. circuit parameters
        using the parameter‑shift rule."""
        grad_results: List[List[float]] = []
        shift = np.pi / 2
        for values in parameter_sets:
            grad_row: List[float] = []
            for i, _ in enumerate(self._parameters):
                plus = list(values)
                minus = list(values)
                plus[i] += shift
                minus[i] -= shift
                exp_plus = self._expectation_value(observable, plus)
                exp_minus = self._expectation_value(observable, minus)
                grad = (exp_plus - exp_minus) / 2.0
                grad_row.append(float(grad))
            grad_results.append(grad_row)
        return grad_results


__all__ = ["FastBaseEstimator"]
