from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers import Backend

class HybridSelfAttention:
    """Quantum circuit that fuses self‑attention style rotations with a QCNN ansatz."""
    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _conv_block(self, qc: QuantumCircuit, q1: int, q2: int) -> QuantumCircuit:
        """A minimal two‑qubit convolution block."""
        qc.cx(q1, q2)
        qc.h(q1)
        qc.cx(q2, q1)
        return qc

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)

        # Self‑attention style rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # QCNN‑style convolution layers
        for i in range(0, self.n_qubits - 1, 2):
            circuit = self._conv_block(circuit, i, i + 1)

        # Entangling CRX gates (pooling analogue)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the hybrid circuit.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Quantum backend to use.
        rotation_params : np.ndarray
            Parameters for the rotation gates; length must be 3 * n_qubits.
        entangle_params : np.ndarray
            Parameters for the CRX entangling gates; length must be n_qubits - 1.
        shots : int, optional
            Number of shots for simulation.

        Returns
        -------
        dict
            Measurement counts dictionary.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

def HybridSelfAttention_factory() -> HybridSelfAttention:
    return HybridSelfAttention()

__all__ = ["HybridSelfAttention", "HybridSelfAttention_factory"]
