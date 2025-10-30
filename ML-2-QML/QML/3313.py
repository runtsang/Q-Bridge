"""Hybrid estimator for quantum circuits with optional shot noise.

The `HybridBaseEstimator` class accepts a Qiskit `QuantumCircuit` and
evaluates expectation values of Pauli observables.  When `shots` is
specified a Qiskit Aer simulator is used to sample measurement outcomes
and compute noisy expectation values; otherwise a deterministic
Statevector evaluation is performed.  A `build_classifier_circuit`
function mirroring the classical builder is also provided.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp


class HybridBaseEstimator:
    """Evaluate quantum circuits with optional shot noise."""

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
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[complex]]:
        """Return expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Pauli operators to measure.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        shots : int | None, optional
            If provided, use a QASM simulator to generate noisy samples.
        seed : int | None, optional
            Random seed for reproducible sampling.
        """
        observables = list(observables)
        results: list[list[complex]] = []

        if shots is None:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(observable) for observable in observables]
                results.append(row)
            return results

        backend = Aer.get_backend("qasm_simulator")
        for values in parameter_sets:
            bound = self._bind(values)
            bound.measure_all()
            job = execute(bound, backend=backend, shots=shots, seed_simulator=seed)
            counts = job.result().get_counts()
            row = []
            for obs in observables:
                pauli_str = obs.to_label()
                exp_val = 0.0
                for bitstring, count in counts.items():
                    parity = sum(
                        int(bitstring[-i - 1])
                        for i, char in enumerate(pauli_str)
                        if char == "Z"
                    )
                    eigen = 1.0 if parity % 2 == 0 else -1.0
                    exp_val += eigen * count
                exp_val /= shots
                row.append(complex(exp_val))
            results.append(row)
        return results


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Parametrized circuit ready for evaluation.
    encoding : Iterable
        Parameter vector for data encoding.
    weights : Iterable
        Parameter vector for variational weights.
    observables : list[SparsePauliOp]
        Pauli observables for expectation value measurement.
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


__all__ = ["HybridBaseEstimator", "build_classifier_circuit"]
