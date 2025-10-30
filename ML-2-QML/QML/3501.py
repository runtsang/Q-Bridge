"""Quantum feature mapper and hybrid estimator for use with the hybrid model.

The implementation uses Qiskit for circuit construction and a
state‑vector simulator for expectation estimation.  It mirrors the
original `build_classifier_circuit` but introduces parameter‑shift
entanglement and a multi‑qubit measurement strategy.  The `EstimatorQNN`
class provides a quantum neural network that can be used as a
regressor, extending the original one‑qubit implementation to a
three‑qubit ansatz.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


# ----------------------------------------------------------------------
# Quantum classifier circuit factory
# ----------------------------------------------------------------------
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz with data‑uploading and
    entangling gates.  The circuit is compatible with the
    classical interface: it returns the circuit, the list of
    encoding and weight parameters, and a set of Z‑type observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data‑uploading layer
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # Variational blocks
    idx = 0
    for _ in range(depth):
        # Rotation layer
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        # Entangling layer
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Measure observables
    observables = [
        SparsePauliOp.from_list(
            [(f"I" * i + "Z" + f"I" * (num_qubits - i - 1), 1)]
        )
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables


# ----------------------------------------------------------------------
# Quantum feature mapper
# ----------------------------------------------------------------------
class QuantumFeatureMapper:
    """
    Evaluates the circuit for a batch of classical inputs and returns a
    numpy array of expectation values.  It uses the Aer state‑vector
    simulator for exact results, but can be swapped for a noisy backend
    if desired.
    """

    def __init__(self, num_qubits: int, depth: int, backend: object | None = None) -> None:
        self.qc, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Args:
            inputs: shape (batch, num_qubits)
        Returns:
            expectations: shape (batch, num_qubits)
        """
        expectations = []
        for sample in inputs:
            bound_qc = self.qc.bind_parameters(dict(zip(self.encoding, sample)))
            job = execute(bound_qc, self.backend, shots=1024, memory=False)
            result = job.result()
            sv = result.get_statevector(bound_qc)
            exp_vals = [pauli.expectation_value(sv).real for pauli in self.observables]
            expectations.append(exp_vals)
        return np.array(expectations)


# ----------------------------------------------------------------------
# Quantum EstimatorQNN
# ----------------------------------------------------------------------
class EstimatorQNN:
    """
    Quantum neural network for regression, extending the one‑qubit
    circuit in the seed.  The ansatz now uses a 3‑qubit
    parameter‑shift architecture with a single Z observable acting on
    the last qubit.
    """

    def __init__(self, backend: object | None = None) -> None:
        self.qc = QuantumCircuit(3)

        # Input encoding
        self.input_params = [ParameterVector("x", 3)]
        for i, p in enumerate(self.input_params[0]):
            self.qc.h(i)
            self.qc.ry(p, i)

        # Parameterised rotation block
        self.weight_params = ParameterVector("w", 9)
        idx = 0
        for i in range(3):
            self.qc.rx(self.weight_params[idx], i)
            idx += 1

        # Entanglement
        self.qc.cz(0, 1)
        self.qc.cz(1, 2)

        # Observable on last qubit
        self.observable = SparsePauliOp.from_list([("Y" * 3, 1)])
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Args:
            inputs: shape (batch, 3)
        Returns:
            predictions: shape (batch, 1)
        """
        preds = []
        for sample in inputs:
            bound_qc = self.qc.bind_parameters(
                {self.input_params[0][i]: sample[i] for i in range(3)}
            )
            job = execute(bound_qc, self.backend, shots=1024)
            result = job.result()
            sv = result.get_statevector(bound_qc)
            pred = self.observable.expectation_value(sv).real
            preds.append([pred])
        return np.array(preds)


__all__ = [
    "build_classifier_circuit",
    "QuantumFeatureMapper",
    "EstimatorQNN",
]
