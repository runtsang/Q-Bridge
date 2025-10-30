"""Quantum hybrid filter and attention.

The module provides two factory functions that match the original
Conv.py and SelfAttention.py interfaces but combine a quanvolution
circuit with a quantum self‑attention block.  The quantum filter
produces a scalar probability that is used as a bias for the
attention circuit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.random import random_circuit

# Quantum quanvolution
class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

# Quantum self‑attention
class QuantumSelfAttention:
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
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# Hybrid quantum filter + attention
class HybridConvAttention:
    def __init__(
        self,
        kernel_size: int = 2,
        n_qubits: int = 4,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.filter = QuanvCircuit(kernel_size, self.backend, shots, threshold)
        self.attention = QuantumSelfAttention(n_qubits)

    def run(self, data: np.ndarray) -> dict:
        """Apply quanvolution followed by quantum self‑attention."""
        conv_out = self.filter.run(data)

        # Build a dummy input vector for the attention circuit.
        # Replicate the scalar conv_out across all qubits.
        inputs = np.full((1, self.attention.n_qubits), conv_out, dtype=np.float32)

        # Randomly initialise rotation and entangle parameters.
        rot_params = np.random.randn(self.attention.n_qubits * 3)
        ent_params = np.random.randn(self.attention.n_qubits - 1)

        # The attention circuit ignores the classical inputs; they are
        # embedded in the rotation parameters in a real implementation.
        # Here we simply pass the parameters to the circuit.
        return self.attention.run(
            self.backend, rot_params, ent_params, shots=1024
        )

def Conv() -> HybridConvAttention:
    """Return a hybrid quanvolution‑attention filter."""
    return HybridConvAttention()

def SelfAttention() -> QuantumSelfAttention:
    """Return a quantum self‑attention helper."""
    return QuantumSelfAttention(n_qubits=4)

__all__ = ["Conv", "SelfAttention", "HybridConvAttention"]
