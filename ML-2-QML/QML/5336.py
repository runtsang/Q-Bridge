"""Quantum auto‑encoder and kernel implementation using Qiskit."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

class QuantumAutoencoder:
    """Variational quantum auto‑encoder using RealAmplitudes and swap test."""
    def __init__(self, latent_dim: int, trash_dim: int = 2, reps: int = 5, backend=None):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs the full auto‑encoder circuit."""
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent part
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=self.reps)
        qc.compose(ansatz, list(range(self.latent_dim + self.trash_dim)), inplace=True)

        # Swap test
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def run(self, shots: int = 1024) -> dict:
        """Execute the circuit on the chosen backend."""
        job = execute(self.circuit, self.backend, shots=shots)
        return job.result().get_counts(self.circuit)

class QuantumKernel:
    """Quantum kernel using RealAmplitudes ansatz and statevector overlap."""
    def __init__(self, num_qubits: int, reps: int = 5, backend=None):
        self.num_qubits = num_qubits
        self.reps = reps
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.ansatz = RealAmplitudes(num_qubits, reps=reps)

    def encode(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the kernel value k(x, y) = |<ψ_x|ψ_y>|^2."""
        qc_x = QuantumCircuit(self.num_qubits)
        for i, val in enumerate(x):
            qc_x.ry(val, i)
        qc_x.compose(self.ansatz, range(self.num_qubits), inplace=True)

        qc_y = QuantumCircuit(self.num_qubits)
        for i, val in enumerate(y):
            qc_y.ry(val, i)
        qc_y.compose(self.ansatz, range(self.num_qubits), inplace=True)

        sv_x = Statevector.from_instruction(qc_x)
        sv_y = Statevector.from_instruction(qc_y)
        overlap = np.abs(np.vdot(sv_x.data, sv_y.data)) ** 2
        return float(overlap)

def quantum_autoencoder_circuit(latent_dim: int, trash_dim: int = 2, reps: int = 5) -> QuantumCircuit:
    """Return a Qiskit circuit implementing a quantum auto‑encoder."""
    ae = QuantumAutoencoder(latent_dim, trash_dim, reps)
    return ae.circuit

def quantum_kernel_matrix(xs: Iterable[np.ndarray], ys: Iterable[np.ndarray], num_qubits: int, reps: int = 5) -> np.ndarray:
    """Compute the Gram matrix between two sets of vectors using a quantum kernel."""
    kernel = QuantumKernel(num_qubits, reps)
    xs = list(xs)
    ys = list(ys)
    return np.array([[kernel.encode(x, y) for y in ys] for x in xs])

__all__ = [
    "QuantumAutoencoder",
    "QuantumKernel",
    "quantum_autoencoder_circuit",
    "quantum_kernel_matrix",
]
