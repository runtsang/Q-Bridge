from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridBaseEstimator:
    """Hybrid estimator that wraps a Qiskit circuit, optionally with a self‑attention style block."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = qiskit.Aer.get_backend("qasm_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate expectation values for each parameter set.

        Args:
            observables: Quantum operators for which expectation values are requested.
            parameter_sets: Sequence of parameter vectors.
            shots: Optional number of shots for sampling instead of exact statevector.

        Returns:
            A list of rows, each containing the expectation values for the provided observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            if shots is None:
                state = Statevector.from_instruction(self._bind(params))
                row = [state.expectation_value(obs) for obs in observables]
            else:
                circuit = self._bind(params)
                job = qiskit.execute(circuit, self._backend, shots=shots)
                counts = job.result().get_counts(circuit)
                # Convert counts to probabilities and compute expectation values
                probs = {k: v / shots for k, v in counts.items()}
                row = [self._expectation_from_counts(obs, probs) for obs in observables]
            results.append(row)

        return results

    def _expectation_from_counts(self, operator: BaseOperator, probs: dict[str, float]) -> complex:
        """Compute expectation value from measurement probabilities."""
        exp = 0.0 + 0.0j
        for bitstring, p in probs.items():
            state = Statevector.from_label(bitstring)
            exp += p * state.expectation_value(operator)
        return exp


__all__ = ["HybridBaseEstimator"]
