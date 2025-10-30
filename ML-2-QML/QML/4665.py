"""Hybrid Qiskit auto‑encoder.

Implements a variational circuit that encodes the input state,
compresses it into a smaller number of qubits (latent space), and
reconstructs the full state.  The circuit is built from a
RealAmplitudes ansatz and a swap‑test style extraction of the latent
qubits.  A SamplerQNN wrapper is provided for easy integration with
Qiskit‑Machine‑Learning optimisers.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumAutoencoder:
    """
    Variational quantum auto‑encoder.

    Parameters
    ----------
    input_qubits : int
        Number of qubits representing the raw input.
    latent_qubits : int
        Size of the compressed latent space.
    reps : int, optional
        Repetitions of the RealAmplitudes ansatz.
    """
    def __init__(self, input_qubits: int, latent_qubits: int, reps: int = 3):
        self.input_qubits = input_qubits
        self.latent_qubits = latent_qubits
        self.reps = reps
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.input_qubits, "q")
        cr = ClassicalRegister(self.input_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # ---------- Encoding ----------
        qc.h(qr)  # simple embedding
        qc.append(RealAmplitudes(self.input_qubits, reps=self.reps), qr)
        qc.barrier()

        # ---------- Latent extraction ----------
        for i in range(self.latent_qubits):
            qc.swap(qr[i], qr[self.input_qubits - self.latent_qubits + i])
        qc.barrier()

        # ---------- Decoding ----------
        qc.append(RealAmplitudes(self.input_qubits, reps=self.reps), qr)
        qc.barrier()
        qc.measure(qr, cr)

        return qc

    def sampler_qnn(self) -> SamplerQNN:
        """
        Wrap the circuit in a SamplerQNN for integration with
        Qiskit‑Machine‑Learning optimisers.
        """
        sampler = Sampler()
        return SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=self.latent_qubits,
            sampler=sampler,
        )

def train_quantum_autoencoder(
    qnn: SamplerQNN,
    data: np.ndarray,
    *,
    epochs: int = 50,
    learning_rate: float = 0.01,
) -> list[float]:
    """
    Very light‑weight training loop that updates the circuit parameters
    using the gradient‑descent optimiser provided by Qiskit‑Machine‑Learning.
    """
    from qiskit_machine_learning.optimizers import GradientDescent

    optimizer = GradientDescent(learning_rate=learning_rate, maxiter=epochs)
    history: list[float] = []

    for _ in range(epochs):
        loss = qnn.loss(data)
        optimizer.step()
        history.append(loss)

    return history


__all__ = [
    "QuantumAutoencoder",
    "train_quantum_autoencoder",
]
