"""Enhanced lightweight estimator utilities built on Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimatorEnhanced:
    """Evaluate expectation values of observables for a parametrized circuit.

    Supports batched evaluation and optional shot‑based Gaussian noise to
    emulate finite‑shot sampling.  The estimator is compatible with
    scikit‑learn style APIs via a ``predict`` method.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        shots: int | None = None,
        backend=None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expectation(self, state: Statevector, observable: BaseOperator) -> complex:
        """Compute expectation value using the statevector."""
        return state.expectation_value(observable)

    def predict(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return a matrix of shape (n_samples, n_observables) with
        computed expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [self._expectation(state, obs) for obs in observables]
            results.append(row)

        array = np.array(results, dtype=complex)

        # If shots are requested, add Gaussian noise to emulate finite‑shot sampling.
        if self.shots is not None:
            rng = np.random.default_rng()
            array = rng.normal(loc=array, scale=1.0 / np.sqrt(self.shots))

        return array

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compatibility wrapper that returns a plain Python list."""
        return self.predict(observables, parameter_sets).tolist()


__all__ = ["FastBaseEstimatorEnhanced"]
