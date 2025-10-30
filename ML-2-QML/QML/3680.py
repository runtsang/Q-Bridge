"""Quantum estimator for parameterized quantum circuits using Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterized circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        return self._circuit.assign_parameters(dict(zip(self._parameters, values)), inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if not observables:
            observables = [BaseOperator.from_label("I")]
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["FastBaseEstimator"]
