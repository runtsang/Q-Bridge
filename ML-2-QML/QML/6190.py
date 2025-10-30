"""Quantum self‑attention using variational circuits.

The circuit encodes the input via angle embedding, applies parameterized rotations
and a linear entangling layer, and extracts attention‑like scores from the
Pauli‑Z expectation values of each qubit.

The public `run` method follows the same signature as the classical version:
    run(backend, rotation_params, entangle_params, inputs, shots=1024)
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import AngleEmbedding
from qiskit.providers.aer import AerSimulator


class SelfAttention:
    """
    Variational self‑attention circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 1).
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """
        Construct a circuit that encodes the input and applies a variational block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for RY gates. Shape: (n_qubits,)
        entangle_params : np.ndarray
            Parameters for CRX entangling gates. Shape: (n_qubits - 1,)
        inputs : np.ndarray
            Input vector to be angle‑embedded. Shape: (n_qubits,)

        Returns
        -------
        QuantumCircuit
            Fully constructed circuit ready for execution.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # 1. Encode the classical input via angle embedding
        embedding = AngleEmbedding(inputs, self.qr, rotation="ry")
        circuit.append(embedding, self.qr)

        # 2. Parameterized rotation layer
        for i, angle in enumerate(rotation_params):
            circuit.ry(angle, self.qr[i])

        # 3. Entangling layer (linear chain of CRX gates)
        for i, theta in enumerate(entangle_params):
            circuit.crx(theta, self.qr[i], self.qr[i + 1])

        # 4. Measure all qubits
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the circuit and return attention‑like scores.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Backend to run the circuit on.
        rotation_params : np.ndarray
            RY rotation angles. Shape: (n_qubits,)
        entangle_params : np.ndarray
            CRX entanglement angles. Shape: (n_qubits - 1,)
        inputs : np.ndarray
            Input vector for angle embedding. Shape: (n_qubits,)
        shots : int, default=1024
            Number of shots for simulation.

        Returns
        -------
        list[float]
            Pauli‑Z expectation values for each qubit, interpreted as attention
            scores in the range [-1, 1].
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Compute expectation of Z for each qubit
        expectations = []
        for qubit in range(self.n_qubits):
            exp_z = 0.0
            for bitstring, freq in counts.items():
                # Bitstring is in reverse order relative to qubit indices
                bit = int(bitstring[::-1][qubit])
                exp_z += (1 - 2 * bit) * freq
            exp_z /= shots
            expectations.append(exp_z)
        return expectations


# Default backend and instance
backend = AerSimulator()
attention = SelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
