"""Quantum estimator with batched parameter sets and optional shot noise using Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator__gen300:
    """Estimator for parametrized quantum circuits with batched evaluation."""

    def __init__(self, circuit: QuantumCircuit, backend: str | None = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or "aer_simulator_statevector"

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for param_set in parameter_sets:
            circ = self._bind(param_set)
            if shots is None:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(circ, backend=Aer.get_backend(self.backend), shots=shots)
                result = job.result()
                counts = result.get_counts(circ)
                state = Statevector.from_counts(counts)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_noisy(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
    ) -> List[List[complex]]:
        """Convenience wrapper that forces shot-based evaluation."""
        return self.evaluate(observables, parameter_sets, shots=shots)


__all__ = ["FastBaseEstimator__gen300"]
