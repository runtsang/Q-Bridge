"""Quantum implementation of a hybrid autoencoder for feature extraction.

The class mirrors the classical `HybridAutoencoderClassifier` but focuses on
the quantum circuit that performs a swap‑test based autoencoding of a
latent vector.  It exposes a `run` method that returns a fidelity value
for a given latent input, enabling integration with a PyTorch training
loop via a differentiable wrapper.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class HybridAutoencoderClassifier:
    """Quantum autoencoder circuit wrapped in a SamplerQNN."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=1,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode part
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap‑test
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _interpret(self, x: np.ndarray) -> np.ndarray:
        """Interpret sampler output as fidelity (0‑1)."""
        return x

    def run(self, latent: np.ndarray) -> np.ndarray:
        """Run the autoencoder circuit for a batch of latent vectors."""
        return self.qnn.forward(latent)

__all__ = ["HybridAutoencoderClassifier"]
