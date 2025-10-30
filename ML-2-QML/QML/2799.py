"""Hybrid estimator module – quantum implementation using Qiskit.

This module defines :class:`FastHybridEstimator` that evaluates a
parameterised QuantumCircuit by constructing a Statevector for each
parameter set and returning expectation values of a list of
``BaseOperator`` observables.  It also exposes
:meth:`build_classifier_circuit` to construct a simple layered ansatz
with explicit encoding and variational parameters, matching the
classical interface.

The estimator is deterministic; shot‑sampling can be added by
wrapping the circuit with a simulator that performs measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import SparsePauliOp


class FastHybridEstimator:
    """Evaluate expectation values of observables for a parameterised circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> list[list[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: list[list[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        return results


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    depth
        Number of variational layers.

    Returns
    -------
    Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]
        The constructed circuit, encoding parameters, variational parameters,
        and a list of Pauli observables (Z on each qubit).
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX rotations
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["FastHybridEstimator", "build_classifier_circuit"]
