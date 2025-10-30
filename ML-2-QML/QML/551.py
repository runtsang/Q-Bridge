"""Quantum autoencoder using a variational circuit and swap‑test for latent extraction."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class AutoencoderHybridQNN(SamplerQNN):
    """A QNN that implements a quantum auto‑encoder with a swap‑test."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 3,
        sampler: Sampler | None = None,
        *,
        output_shape: int | None = None,
    ) -> None:
        if sampler is None:
            sampler = Sampler()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        circuit = self._build_circuit()
        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=output_shape or 2,
            sampler=sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs the auto‑encoder circuit with a swap‑test."""
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash subspace
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.append(ansatz, range(0, self.num_latent + self.num_trash))

        # Swap‑test auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean‑squared error loss between predicted and target probability."""
        return float(np.mean((predictions - targets) ** 2))


def train_qnn(
    qnn: AutoencoderHybridQNN,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    optimizer_cls=COBYLA,
    seed: int | None = 42,
) -> list[float]:
    """Trains the quantum auto‑encoder via a classical optimizer."""
    algorithm_globals.random_seed = seed or 0
    optimizer = optimizer_cls(maxiter=epochs, tol=1e-6, disp=False)
    history: list[float] = []

    # Prepare target: ideal reconstruction is 1 on auxiliary qubit for identical states
    targets = np.ones((len(data), 1))

    def objective(params: np.ndarray) -> float:
        preds = qnn.forward(params)
        loss = qnn.loss(preds, targets)
        return loss

    for _ in range(epochs):
        params, loss_val, _ = optimizer.optimize(
            num_vars=len(qnn.weight_params), objective_function=objective, initial_point=np.random.rand(len(qnn.weight_params))
        )
        history.append(loss_val)
    return history


__all__ = ["AutoencoderHybridQNN", "train_qnn"]
