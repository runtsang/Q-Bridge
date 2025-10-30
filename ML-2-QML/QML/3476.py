"""Hybrid classifier with Qiskit simulation.

This module defines `HybridFCL`, a class that builds a parameterised
quantum circuit in Qiskit, runs it on a state‑vector simulator,
and returns expectation values of Pauli‑Z on each qubit.  The
class is a drop‑in replacement for the classical version and
provides a quantum‑centric contribution.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridFCL:
    """
    Hybrid classifier that builds a Qiskit circuit with:
      • Rx encoding gates mapped to input features,
      • Ry variational gates with learnable parameters,
      • CZ entangling gates between neighbouring qubits.

    The circuit is executed on a state‑vector simulator and the
    expectation values of Pauli‑Z on each qubit are returned.
    """

    def __init__(self, num_features: int, n_qubits: int, depth: int, num_classes: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.num_classes = num_classes

        # Parameter vectors
        self.encoding = ParameterVector('x', num_qubits)
        self.weights = ParameterVector('theta', num_qubits * depth)

        # Build the circuit
        self.circuit = QuantumCircuit(n_qubits)
        for i, param in enumerate(self.encoding):
            self.circuit.rx(param, i)

        idx = 0
        for _ in range(depth):
            for i in range(n_qubits):
                self.circuit.ry(self.weights[idx], i)
                idx += 1
            for i in range(n_qubits - 1):
                self.circuit.cz(i, i + 1)

        # Observables: Pauli‑Z on each qubit
        self.observables = [SparsePauliOp(f"{'I'*i}Z{'I'*(n_qubits-i-1)}") for i in range(n_qubits)]

        # Backend
        self.backend = Aer.get_backend('statevector_simulator')

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of input samples.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, num_features) containing raw features.

        Returns
        -------
        np.ndarray
            Array of shape (batch, n_qubits) with expectation values of Z.
        """
        batch = inputs.shape[0]
        expectations = np.zeros((batch, self.n_qubits))

        for i in range(batch):
            # Bind encoding parameters to feature values
            bindings = {self.encoding[j]: inputs[i, j] for j in range(self.n_qubits)}
            bound_circ = self.circuit.bind_parameters(bindings)

            job = execute(bound_circ, self.backend)
            result = job.result()
            statevector = result.get_statevector(bound_circ)

            for q in range(self.n_qubits):
                expectations[i, q] = self._expectation(statevector, q)

        return expectations

    def _expectation(self, statevector: np.ndarray, qubit: int) -> float:
        """
        Compute expectation of Z on a given qubit from the statevector.
        """
        exp = 0.0
        for idx, amp in enumerate(statevector):
            bit = (idx >> qubit) & 1
            prob = abs(amp) ** 2
            exp += (1.0 if bit == 0 else -1.0) * prob
        return exp

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Simple linear readout on quantum expectations.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input features of shape (batch, num_features).

        Returns
        -------
        np.ndarray
            Logits of shape (batch, num_classes).
        """
        quantum_features = self.run(inputs)
        # Random linear layer (placeholder for an actual trainable head)
        W = np.random.randn(self.n_qubits, self.num_classes)
        b = np.random.randn(self.num_classes)
        logits = quantum_features @ W + b
        return logits

__all__ = ["HybridFCL"]
