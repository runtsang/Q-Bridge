"""Quantum self‑attention circuit using Qiskit.

The class mirrors the classical interface but maps the attention
operations to a parameterized variational circuit.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """
    Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, one per head in the classical counterpart.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a parameterized circuit.

        rotation_params : np.ndarray
            Shape (n_qubits, 3) – RX, RY, RZ for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – CX rotation angles for adjacent qubits.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply rotations
        for i in range(self.n_qubits):
            rx, ry, rz = rotation_params[i]
            circuit.rx(rx, i)
            circuit.ry(ry, i)
            circuit.rz(rz, i)

        # Entanglement layer (controlled‑RX)
        for i in range(self.n_qubits - 1):
            theta = entangle_params[i]
            circuit.crx(theta, i, i + 1)

        # Measure all qubits
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the circuit on the supplied backend.

        Returns
        -------
        dict
            Measurement counts mapping bitstring → frequency.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)

# Default backend for convenience
backend = Aer.get_backend("qasm_simulator")

__all__ = ["SelfAttention"]
