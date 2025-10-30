"""Hybrid quantum autoencoder using Qiskit and a sampler-based QNN."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def _create_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode data into the first num_latent qubits
    qc.append(RealAmplitudes(num_latent, reps=3), range(num_latent))

    # Add trash qubits and a swap test
    for i in range(num_trash):
        qc.cx(num_latent + i, num_latent + num_trash + i)

    # Auxiliary qubit for swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

class QAutoencoder:
    """Quantum autoencoder with a variational circuit and a classical loss."""

    def __init__(self, num_latent: int = 3, num_trash: int = 2, seed: int = 42):
        algorithm_globals.random_seed = seed
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.sampler = Sampler()
        self.circuit = _create_autoencoder_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.optimizer = COBYLA(maxiter=200)

    def loss(self, params: np.ndarray, target: np.ndarray) -> float:
        """Compute the mean squared error between the QNN output and target."""
        output = np.array(self.qnn.predict(params))
        return np.mean((output - target) ** 2)

    def train(self, targets: np.ndarray, *, epochs: int = 50, lr: float = 0.01) -> list[float]:
        """Train the quantum autoencoder using a classical optimizer."""
        history: list[float] = []
        params = np.random.uniform(0, 2 * np.pi, size=len(self.circuit.parameters))
        for _ in range(epochs):
            loss_val = self.loss(params, targets)
            history.append(loss_val)
            params = self.optimizer.minimize(lambda p: self.loss(p, targets), params)
        return history

__all__ = ["QAutoencoder", "_create_autoencoder_circuit"]
