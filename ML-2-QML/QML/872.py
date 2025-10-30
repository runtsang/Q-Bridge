"""Quantum autoencoder implemented with Pennylane.

This module implements a hybrid variational autoencoder that compresses a classical
vector into a low‑dimensional quantum latent space and reconstructs it.  The circuit
uses a RealAmplitudes ansatz, a swap‑test based similarity measure and a gradient
optimiser.  The API mirrors the classical counterpart for easy comparison.
"""

import pennylane as qml
import numpy as np
from typing import Tuple, List

class AutoencoderNet:
    """Quantum variational autoencoder."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int = 3,
        reps: int = 3,
        device_name: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.reps = reps
        self.device = qml.device(device_name, wires=num_qubits)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="numpy")
        def circuit(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode classical data into qubits via RealAmplitudes
            qml.templates.RealAmplitudes(x, wires=range(self.num_qubits))
            # Variational layer on latent qubits
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.latent_dim))
            # Decode: apply another ansatz on all qubits
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        self.circuit = circuit
        # Initialise parameters: two blocks (encoder & decoder)
        self.params = np.random.randn(2, self.reps, self.num_qubits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the expectation values for the given input."""
        return self.circuit(x, self.params)

    def loss(self, x: np.ndarray, recon: np.ndarray) -> float:
        return np.mean((x - recon) ** 2)

    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> List[float]:
        opt = qml.AdamOptimizer(stepsize=lr)
        history: List[float] = []
        for epoch in range(epochs):
            def cost(p):
                total = 0.0
                for x in data:
                    recon = self.circuit(x, p)
                    total += self.loss(x, recon)
                return total / len(data)
            self.params = opt.step(cost, self.params)
            history.append(cost(self.params))
        return history

    def sample(self, num_samples: int) -> np.ndarray:
        """Generate random inputs and decode."""
        samples = np.random.randn(num_samples, self.num_qubits)
        return np.array([self.circuit(s, self.params) for s in samples])

def Autoencoder(
    num_qubits: int,
    *,
    latent_dim: int = 3,
    reps: int = 3,
    device_name: str = "default.qubit",
) -> AutoencoderNet:
    """Factory mirroring the classical API."""
    return AutoencoderNet(
        num_qubits=num_qubits,
        latent_dim=latent_dim,
        reps=reps,
        device_name=device_name,
    )

__all__ = ["Autoencoder", "AutoencoderNet"]
