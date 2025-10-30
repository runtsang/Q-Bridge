"""Hybrid quantum autoencoder leveraging a RealAmplitudes ansatz, swap test similarity and a state‑vector kernel."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


class HybridAutoencoderQNN:
    """
    Quantum autoencoder that learns a latent representation through a parameterized ansatz.
    The circuit is composed of:
        * a data‑encoding RealAmplitudes ansatz,
        * a swap‑test that compares the encoded state with a reference,
        * an optional kernel evaluation via state‑vector overlap.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        reps: int = 5,
        backend: str | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.backend = Aer.get_backend(backend or "qasm_simulator")
        self.sampler = Aer.get_backend("statevector_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full auto‑encoder circuit with a swap‑test."""
        total_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=self.reps)
        qc.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        qc.barrier()

        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode a batch of classical data vectors into a similarity score
        obtained from the swap‑test.
        """
        scores = []
        for vec in data:
            qc = self.circuit.copy()
            # Simple amplitude encoding: flip qubits where the feature is 1
            for i, bit in enumerate(vec):
                if bit > 0.5:
                    qc.x(i)
            job = execute(qc, self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts()
            scores.append(counts.get("1", 0) / 1024)
        return np.array(scores)

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the quantum kernel matrix via state‑vector overlap."""
        def state_vector(vec: np.ndarray) -> np.ndarray:
            qc = QuantumCircuit(self.latent_dim + self.trash_dim)
            for i, bit in enumerate(vec):
                if bit > 0.5:
                    qc.x(i)
            job = execute(qc, self.sampler)
            return job.result().get_statevector()

        mat = np.zeros((len(a), len(b)))
        for i, va in enumerate(a):
            sv_a = state_vector(va)
            for j, vb in enumerate(b):
                sv_b = state_vector(vb)
                mat[i, j] = np.abs(np.vdot(sv_a, sv_b)) ** 2
        return mat

    def build_qnn(self) -> SamplerQNN:
        """Instantiate a SamplerQNN that can be trained with a classical optimiser."""
        def identity(x):
            return x

        qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=identity,
            output_shape=1,
            sampler=self.sampler,
        )
        return qnn

    def train(self, data: np.ndarray, labels: np.ndarray, epochs: int = 50):
        """
        Simple training loop that optimises the ansatz parameters using COBYLA.
        """
        qnn = self.build_qnn()
        opt = COBYLA(maxiter=epochs * 10)
        opt.minimize(
            fun=lambda p: self.loss(p, data, labels),
            x0=np.random.rand(len(list(self.circuit.parameters))),
        )
        return opt

    def loss(self, params: np.ndarray, data: np.ndarray, labels: np.ndarray) -> float:
        """Mean‑squared error between predicted and target similarity."""
        self.circuit.assign_parameters(params, inplace=True)
        preds = self.encode(data)
        return np.mean((preds - labels) ** 2)


# Alias for unified API
HybridAutoencoder = HybridAutoencoderQNN

__all__ = ["HybridAutoencoderQNN"]
