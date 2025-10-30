"""
Quantum self‑attention block with parameterised rotations and entanglement.
Supports both state‑vector simulation for hybrid feature extraction and
classical sampling for pure quantum inference.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import torch


class SelfAttentionEnhanced:
    """
    Quantum self‑attention circuit that mirrors the classical interface.
    rotation_params and entangle_params are flattened arrays that set the
    parameters of the RX, RY, RZ rotation gates and CX‑based entanglement.
    """
    def __init__(self, n_qubits: int = 4, use_statevector: bool = False):
        self.n_qubits = n_qubits
        self.use_statevector = use_statevector
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer (controlled‑RX)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(self, backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024,
            return_statevector: bool = False):
        circuit = self._build_circuit(rotation_params, entangle_params)
        if return_statevector:
            # Return the full statevector for hybrid feature extraction
            backend = Aer.get_backend("statevector_simulator")
            result = execute(circuit, backend).result()
            return result.get_statevector(circuit)
        else:
            circuit.measure(self.qr, self.cr)
            result = execute(circuit, backend, shots=shots).result()
            return result.get_counts(circuit)


def SelfAttention():
    """
    Factory that returns an instance of SelfAttentionEnhanced.
    """
    backend = Aer.get_backend("qasm_simulator")
    return SelfAttentionEnhanced(n_qubits=4, use_statevector=False)
