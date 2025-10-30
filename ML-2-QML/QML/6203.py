"""Hybrid quantum fully connected layer with efficient state‑vector evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridFCLQuantum:
    """Quantum parameterized circuit that returns expectation values efficiently."""

    def __init__(self, circuit: QuantumCircuit, shots: int = 1024, backend=None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the provided values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Quantum operators whose expectation values are required.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        return results


__all__ = ["HybridFCLQuantum"]
