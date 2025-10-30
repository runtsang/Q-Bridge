"""Hybrid quantum estimator that mirrors the classical interface.

Features:
* Unified evaluate interface for Qiskit circuits.
* Optional shot noise simulation using a state‑vector backend.
* Convenience factory for building classifier ansatzes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.sparse_pauli_op import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a simple layered ansatz with metadata.

    The returned tuple mirrors the signature of the classical counterpart:
    * ``circuit`` – the Qiskit circuit.
    * ``encoding`` – list of encoding parameters.
    * ``weights`` – list of variational parameters.
    * ``observables`` – list of Pauli operators with a single Z per qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class FastBaseEstimator:
    """Evaluate expectation values of a parameterised circuit.

    Parameters
    ----------
    circuit
        A :class:`qiskit.circuit.QuantumCircuit` instance that contains
        ``Parameter`` objects.
    """

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
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add classical shot noise to the ideal expectation values."""
        base = self.evaluate(observables, parameter_sets)
        if shots is None:
            return base
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in base:
            noisy_row = [
                rng.normal(z.real, max(1e-6, 1 / shots))
                + 1j * rng.normal(z.imag, max(1e-6, 1 / shots))
                for z in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "build_classifier_circuit"]
