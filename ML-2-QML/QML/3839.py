"""Hybrid estimator for Qiskit circuits with optional shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """Evaluate parametrized quantum circuits with optional shot noise."""

    def __init__(self, circuit: QuantumCircuit, *, shots: Optional[int] = None, seed: Optional[int] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]

            if self.shots is not None:
                rng = np.random.default_rng(self.seed)
                std = max(1e-6, 1 / self.shots)
                row = [float(rng.normal(complex(val).real, std)) for val in row]

            results.append(row)
        return results


__all__ = ["FastHybridEstimator"]
