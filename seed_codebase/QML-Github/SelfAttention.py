"""Quantum self-attention built solely with Qiskit."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def SelfAttention():
    class QuantumSelfAttention:
        """Basic quantum circuit representing a self-attention style block."""

        def __init__(self, n_qubits: int):
            self.n_qubits = n_qubits
            self.qr = QuantumRegister(n_qubits, "q")
            self.cr = ClassicalRegister(n_qubits, "c")

        def _build_circuit(
            self, rotation_params: np.ndarray, entangle_params: np.ndarray
        ) -> QuantumCircuit:
            circuit = QuantumCircuit(self.qr, self.cr)
            for i in range(self.n_qubits):
                circuit.rx(rotation_params[3 * i], i)
                circuit.ry(rotation_params[3 * i + 1], i)
                circuit.rz(rotation_params[3 * i + 2], i)
            for i in range(self.n_qubits - 1):
                circuit.crx(entangle_params[i], i, i + 1)
            circuit.measure(self.qr, self.cr)
            return circuit

        def run(
            self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024,
        ):
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = qiskit.execute(circuit, backend, shots=shots)
            return job.result().get_counts(circuit)
    backend = qiskit.Aer.get_backend("qasm_simulator")
    attention = QuantumSelfAttention(n_qubits=4)
    return attention    
