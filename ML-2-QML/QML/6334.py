"""UniversalEstimator: a hybrid quantum estimator that evaluates expectation values for a parametrized circuit with shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import Parameter


class UniversalEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit with optional shot noise."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compute expectation values for each parameter set and observable. Returns a NumPy array."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs).real for obs in observables]
            results.append(row)
        arr = np.array(results, dtype=np.complex128)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 1 / np.sqrt(shots), size=arr.shape)
            arr += noise
        return arr

__all__ = ["UniversalEstimator"]
