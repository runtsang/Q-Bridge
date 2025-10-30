"""Quantum autoencoder using Qiskit and the StatevectorSampler.

The circuit implements a swap‑test based autoencoder with an optional
domain wall.  It is wrapped in a `SamplerQNN` so it can be trained
with classical optimizers in a hybrid setting.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

# Ensure deterministic behaviour
algorithm_globals.random_seed = 42


class QuantumAutoencoder:
    """Variational autoencoder built from a swap‑test circuit."""

    def __init__(self, num_latent: int, num_trash: int) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=2,
            sampler=Sampler(),
        )

    @staticmethod
    def _ansatz(num_qubits: int) -> QuantumCircuit:
        """Parameterized ansatz for the latent + trash qubits."""
        return RealAmplitudes(num_qubits, reps=5)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash qubits
        qc.compose(
            self._ansatz(self.num_latent + self.num_trash),
            range(0, self.num_latent + self.num_trash),
            inplace=True,
        )
        qc.barrier()

        # Swap test on the auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Run the QNN on classical data.

        Parameters
        ----------
        data : np.ndarray
            Batch of input features (shape: [batch, features]).
            The user is responsible for encoding the data into the
            circuit's input qubits (here we simply assume the data
            is already in the correct format).
        """
        return self.qnn.forward(data)


def create_quantum_autoencoder(num_latent: int = 3, num_trash: int = 2) -> QuantumAutoencoder:
    """Convenience factory mirroring the seed `Autoencoder` helper."""
    return QuantumAutoencoder(num_latent=num_latent, num_trash=num_trash)


__all__ = ["QuantumAutoencoder", "create_quantum_autoencoder"]
