"""Quantum hybrid autoencoder using Qiskit.

This module defines :class:`HybridAutoencoder`, a quantum neural network that
implements an autoencoder circuit with a quantum kernel regularizer.
The encoder and decoder are realized by parameterized quantum circuits
and the kernel is evaluated via a swap‑test style circuit.  The class
provides a classical interface compatible with the PyTorch version.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance


class HybridAutoencoder:
    """
    A quantum autoencoder that uses a RealAmplitudes ansatz for both
    encoder and decoder, and a swap‑test based quantum kernel for
    latent‑space regularization.
    """

    def __init__(self, latent_dim: int = 3, trash_dim: int = 2):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + 2 * trash_dim + 1
        self.sim = AerSimulator()
        self._build_circuits()

    def _build_circuits(self):
        # Encoder and decoder ansatz (mirror of each other)
        self.encoder_circuit = RealAmplitudes(self.latent_dim + self.trash_dim, reps=3)
        self.decoder_circuit = RealAmplitudes(self.latent_dim + self.trash_dim, reps=3)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into latent amplitudes using a swap‑test."""
        n_samples = data.shape[0]
        data_norm = 2 * (data - data.min()) / (data.max() - data.min()) - 1
        encoded = []
        for x in data_norm:
            qc = QuantumCircuit(self.num_qubits)
            qc.initialize(x, list(range(self.latent_dim + self.trash_dim)))
            qc.compose(self.encoder_circuit, inplace=True)
            result = self.sim.run(qc).result()
            state = result.get_statevector()
            encoded.append(state[: self.latent_dim])
        return np.array(encoded)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to data space."""
        decoded = []
        for z in latent:
            qc = QuantumCircuit(self.num_qubits)
            qc.initialize(z, list(range(self.latent_dim)))
            qc.compose(self.decoder_circuit, inplace=True)
            result = self.sim.run(qc).result()
            state = result.get_statevector()
            decoded.append(state[: self.latent_dim])
        return np.array(decoded)

    def forward(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def quantum_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate a swap‑test based quantum kernel between two vectors."""
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(x, list(range(self.latent_dim + self.trash_dim)))
        qc.initialize(y, list(range(self.latent_dim + self.trash_dim, self.latent_dim + 2 * self.trash_dim)))
        qc.h(self.num_qubits - 1)
        for i in range(self.trash_dim):
            qc.cswap(self.num_qubits - 1, i, self.latent_dim + i)
        qc.h(self.num_qubits - 1)
        qc.measure(self.num_qubits - 1, 0)
        job = self.sim.run(qc, shots=1024)
        result = job.result()
        counts = result.get_counts()
        prob0 = counts.get('0', 0) / 1024
        return prob0

    def kernel_regularizer(self, latent: np.ndarray) -> float:
        """Compute kernel regularization over a batch of latent vectors."""
        n = latent.shape[0]
        reg = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                reg += self.quantum_kernel(latent[i], latent[j])
        return reg / (n * (n - 1) / 2)

__all__ = ["HybridAutoencoder"]
