"""Hybrid quantum estimator that evaluates a parameterised circuit with optional Gaussian shot noise."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Iterable, Sequence

class HybridEstimator:
    """A lightweight quantum estimator that can evaluate expectations for a circuit
    and add Gaussian noise to emulate finite shot counts."""
    def __init__(self, circuit: QuantumCircuit, observables: Iterable[BaseOperator]) -> None:
        self._circuit = circuit
        self._observables = list(observables)
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.
        If *shots* is provided, Gaussian noise is added to each expectation value."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in self._observables]
            results.append(row)

        if shots is None:
            return results

        import numpy as np
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridEstimator"]
