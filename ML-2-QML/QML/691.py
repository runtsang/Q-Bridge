"""Enhanced quantum estimator using Qiskit with batched evaluation and shot noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values for a parametrized circuit with batched evaluation
    and optional shot noise simulation."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _statevector(self, bound_circuit: QuantumCircuit) -> Statevector:
        return Statevector.from_instruction(bound_circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of Hermitian operators (Pauli or general).
        parameter_sets
            Sequence of parameter lists to bind to the circuit.
        shots
            If set, Gaussian shot noise with variance 1/shots is added.
        seed
            Random seed for shot noise reproducibility.
        batch_size
            Number of parameter sets processed in a single batch.  If None, all are
            processed at once.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed)

        if batch_size is None or batch_size <= 0:
            batch_size = len(parameter_sets)

        for start in range(0, len(parameter_sets), batch_size):
            batch = parameter_sets[start : start + batch_size]
            for values in batch:
                bound = self._bind(values)
                state = self._statevector(bound)
                row = [state.expectation_value(obs) for obs in observables]
                if shots is not None:
                    row = [val + rng.normal(0.0, 1.0 / np.sqrt(shots)) for val in row]
                results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
