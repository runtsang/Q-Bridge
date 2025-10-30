"""Quantum self‑attention combining convolution, autoencoder, and kernel circuits."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit

class QuantumAutoencoder:
    """
    Quantum autoencoder based on a RealAmplitudes ansatz.
    """
    def __init__(self, num_qubits: int, reps: int = 2):
        self.num_qubits = num_qubits
        self.reps = reps

    def circuit(self, data: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(qr)
        for i, val in enumerate(data):
            if val > 0.5:
                qc.x(qr[i])
        ansatz = RealAmplitudes(self.num_qubits, reps=self.reps)
        qc.append(ansatz, qr)
        return qc

class QuantumConvolution:
    """
    Quantum convolution layer implemented as a random circuit with a threshold.
    """
    def __init__(self, kernel_size: int, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold

    def circuit(self, data: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)
        for i, val in enumerate(data):
            theta = np.pi if val > self.threshold else 0.0
            qc.rx(theta, qr[i])
        qc.barrier()
        qc += random_circuit(self.n_qubits, depth=2)
        return qc

class QuantumSelfAttentionGen240:
    """
    Quantum self‑attention that stitches together convolution and autoencoder circuits.
    """
    def __init__(self, n_qubits: int = 4, conv_kernel_size: int = 2, conv_threshold: float = 0.5) -> None:
        self.n_qubits = n_qubits
        self.conv = QuantumConvolution(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.autoencoder = QuantumAutoencoder(num_qubits=n_qubits, reps=2)

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        data: np.ndarray,
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], qr[i])
            qc.ry(rotation_params[3 * i + 1], qr[i])
            qc.rz(rotation_params[3 * i + 2], qr[i])
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], qr[i], qr[i + 1])
        conv_circ = self.conv.circuit(data)
        qc.compose(conv_circ, qubits=range(self.n_qubits), inplace=True)
        ae_circ = self.autoencoder.circuit(data)
        qc.compose(ae_circ, qubits=range(self.n_qubits), inplace=True)
        qc.measure_all()
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        data: np.ndarray,
        shots: int = 1024,
    ):
        qc = self._build_circuit(rotation_params, entangle_params, data)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

def SelfAttention() -> QuantumSelfAttentionGen240:
    return QuantumSelfAttentionGen240(n_qubits=4, conv_kernel_size=2, conv_threshold=0.5)

__all__ = ["SelfAttention", "QuantumSelfAttentionGen240", "QuantumAutoencoder", "QuantumConvolution"]
