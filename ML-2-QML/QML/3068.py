"""Hybrid quantum autoencoder leveraging a variational circuit with domain‑wall encoding and a swap‑test.

The module defines a `HybridAutoencoder` class that builds a quantum circuit comprising a RealAmplitudes
ansatz, a domain‑wall pattern on the trash qubits, and a swap‑test with an auxiliary qubit.
The returned `SamplerQNN` can be trained with a simple loss function via a COBYLA optimizer.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42


class HybridAutoencoder:
    """Variational autoencoder that learns a latent representation via a swap‑test circuit."""

    def __init__(self, latent_dim: int, trash_dim: int, reps: int = 5) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + first trash block
        qc.compose(
            self._ansatz(self.latent_dim + self.trash_dim),
            range(0, self.latent_dim + self.trash_dim),
            inplace=True,
        )
        qc.barrier()

        # Domain‑wall: flip the second trash block
        for i in range(self.trash_dim):
            qc.x(self.latent_dim + self.trash_dim + i)

        # Swap‑test with auxiliary qubit
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)

        qc.measure(aux, cr[0])
        return qc

    def get_qnn(self, weight_params: list[np.ndarray]) -> SamplerQNN:
        """Return a SamplerQNN instance that maps input parameters to measurement probabilities."""
        def interpret(x: np.ndarray) -> np.ndarray:
            return x

        return SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=weight_params,
            interpret=interpret,
            output_shape=2,
            sampler=self.sampler,
        )


def train_hybrid_autoencoder(
    qnn: SamplerQNN,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    optimizer_cls=COBYLA,
    device: str | None = None,
) -> list[float]:
    """Simple training loop that updates the variational parameters to minimize reconstruction loss."""
    opt = optimizer_cls()
    loss_history: list[float] = []

    for _ in range(epochs):
        def loss_func(params):
            qnn.set_weights(params)
            preds = qnn.predict(data)
            # Reconstruction loss: mean squared error between predicted and target
            loss = np.mean((preds - data) ** 2)
            return loss

        params = opt.optimize(
            num_vars=len(qnn.weights().tolist()),
            objective_function=loss_func,
            initial_point=qnn.weights().tolist(),
            max_iter=200,
        )
        loss_history.append(loss_func(params))
    return loss_history


__all__ = [
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
