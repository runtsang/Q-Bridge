"""Quantum hybrid autoencoder using a QCNN‑inspired variational encoder
and a classical decoder derived from the classical latent representation."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QuantumHybridAutoencoder:
    """Hybrid autoencoder: variational quantum encoder + classical decoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        shots: int = 1024,
        reps: int = 2,
    ) -> None:
        algorithm_globals.random_seed = 42
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.shots = shots

        # Feature map and variational ansatz
        self.feature_map = ZFeatureMap(input_dim, reps=1)
        self.ansatz = RealAmplitudes(input_dim, reps=reps)

        # Estimator for circuit evaluation
        self.estimator = Estimator()

        # QNN that maps classical data to expectation values (latent vector)
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=[SparsePauliOp.from_list([("Z" * input_dim, 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode classical data to a quantum latent representation."""
        # Expectation values from the QNN serve as the latent vector
        return self.qnn.predict(inputs)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent vector back to classical output using a simple
        classical linear transformation (this is a placeholder for a
        more sophisticated decoder)."""
        # For demonstration, we simply return the latent vector
        # In practice, a classical neural network could be attached here.
        return latents

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Full autoencoder forward pass."""
        latent = self.encode(inputs)
        return self.decode(latent)


def QuantumHybridAutoencoderFactory(
    input_dim: int,
    latent_dim: int,
    shots: int = 1024,
    reps: int = 2,
) -> QuantumHybridAutoencoder:
    """Factory returning a configured quantum hybrid autoencoder."""
    return QuantumHybridAutoencoder(input_dim, latent_dim, shots, reps)


__all__ = ["QuantumHybridAutoencoder", "QuantumHybridAutoencoderFactory"]
