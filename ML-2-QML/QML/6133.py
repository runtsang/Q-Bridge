"""Quantum self‑attention using a parameter‑shaped variational circuit.

The circuit applies rotations on each qubit, entangles adjacent qubits,
and then measures the expectation value of the Pauli‑Z operator on each
qubit.  The result is a vector of expectation values that can be fed
into a classical model.  The interface matches the classical version
so that the two can be swapped seamlessly.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttentionModule:
    def __init__(self, n_qubits: int):
        """Create a quantum self‑attention block.

        Parameters
        ----------
        n_qubits : int
            Number of qubits (must match embed_dim in the classical version).
        """
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Build the variational circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for RX, RY, RZ on each qubit.
        entangle_params : np.ndarray
            Parameters for CRX gates between adjacent qubits.

        Returns
        -------
        QuantumCircuit
            The constructed circuit.
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Apply entangling gates
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        # Measure in computational basis
        circuit.measure(self.qr, self.cr)
        return circuit

    def _expectation_from_counts(self, counts: dict, shots: int) -> np.ndarray:
        """Compute expectation values of Pauli‑Z for each qubit.

        Parameters
        ----------
        counts : dict
            Measurement counts.
        shots : int
            Number of shots used.

        Returns
        -------
        np.ndarray
            Vector of expectation values of shape (n_qubits,).
        """
        exp_vals = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            # bitstring is in little‑endian order
            for idx in range(self.n_qubits):
                bit = int(bitstring[self.n_qubits - 1 - idx])  # reverse for little‑endian
                exp_vals[idx] += (1 if bit == 0 else -1) * cnt
        exp_vals /= shots
        return exp_vals

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """Execute the circuit and return expectation values.

        Parameters
        ----------
        backend : qiskit.providers.Provider
            Backend to execute the circuit on.
        rotation_params : np.ndarray
            Rotation parameters.
        entangle_params : np.ndarray
            Entanglement parameters.
        shots : int, optional
            Number of shots.

        Returns
        -------
        np.ndarray
            Expectation values of shape (n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return self._expectation_from_counts(counts, shots)

# Default backend and instance
backend = Aer.get_backend("qasm_simulator")
attention = SelfAttentionModule(n_qubits=4)
