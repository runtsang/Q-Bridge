"""Hybrid estimator for quantum circuits with variational ansatz and shot sampling."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator


class HybridBaseEstimator:
    """Evaluate a parameterised quantum circuit for many parameter sets."""

    def __init__(self, circuit: QuantumCircuit, backend: str | None = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or "statevector_simulator"

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the supplied values."""
        if len(params)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, params))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Return expectation values for each observable and parameter set.

        Parameters
        ----------
        shots:
            If provided, Gaussian noise with variance 1/shots is added to emulate sampling.
        seed:
            Random seed for reproducibility.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    complex(rng.normal(val.real, max(1e-6, 1 / shots))) for val in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Build a variational ansatz with explicit encoding and observables.

    Returns
    -------
    circuit:
        A parameterised ``QuantumCircuit`` ready for training.
    encoding:
        List of input parameters used for data encoding.
    weights:
        List of variational parameters.
    observables:
        Pauli‑Z observables on each qubit.
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
