"""Quantum hybrid convolution‑attention module built on Qiskit.

The quantum implementation reproduces the structure of HybridConvAttention
using a quanvolution circuit followed by a variational self‑attention
circuit.  The `run()` method accepts the same 2‑D data array and returns
an attention‑style probability vector."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.providers import Backend

class QuantumSelfAttention:
    """Variational self‑attention block implemented with a parameter‑ized circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(self, backend: Backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


class QuanvCircuit:
    """Quantum convolution (quanvolution) circuit for a 2‑D filter."""
    def __init__(self, kernel_size: int, backend: Backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Return average probability of measuring |1> across all qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class HybridConvAttention:
    """Quantum analogue of the classical HybridConvAttention module."""
    def __init__(self, kernel_size: int = 2, threshold: float = 127, embed_dim: int = 4):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quanv = QuanvCircuit(kernel_size, self.backend, shots=100, threshold=threshold)
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)

    def run(self, data: np.ndarray, shots: int = 1024) -> dict:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)
            shots: number of shots for the self‑attention circuit

        Returns:
            Counts dictionary from the attention circuit, representing
            the quantum attention output.
        """
        conv_out = self.quanv.run(data)
        # Translate conv_out into rotation and entanglement parameters
        rotation_params = np.full((self.attention.n_qubits * 3,), conv_out, dtype=np.float32)
        entangle_params = np.full((self.attention.n_qubits - 1,), conv_out, dtype=np.float32)
        return self.attention.run(self.backend, rotation_params, entangle_params, shots=shots)


__all__ = ["HybridConvAttention"]
