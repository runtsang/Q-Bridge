"""Quantum self‑attention module.

The circuit implements a parameterised self‑attention block
using single‑qubit rotations and controlled‑RX entanglement.
The output is a vector of expectation values that can be used
as features for downstream quantum or classical models.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the self‑attention circuit and return the
        expectation value of Z on each qubit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(3 * n_qubits,)`` – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape ``(n_qubits-1,)`` – CRX angles between adjacent qubits.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert counts to expectation values of Z
        expectations = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            prob = cnt / self.shots
            bits = np.array([int(b) for b in state[::-1]])  # little‑endian
            z_vals = 1 - 2 * bits  # map 0->1, 1->-1
            expectations += prob * z_vals
        return expectations

__all__ = ["SelfAttention"]
