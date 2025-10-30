"""Quantum autoencoder using PennyLane with parameter‑shift training."""

import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    num_qubits: int
    latent_dim: int
    reps: int = 3
    device: str = "default.qubit"


class Autoencoder__gen406:
    """Variational quantum autoencoder with a latent bottleneck."""

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.dev = qml.device(config.device, wires=config.num_qubits)
        self.latent_wires = list(range(config.latent_dim))
        self.data_wires = list(range(config.latent_dim, config.num_qubits))
        # Random initial parameters
        self.params = np.random.randn(config.reps, config.num_qubits, 3)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Data encoding
            for i, w in enumerate(self.data_wires):
                qml.RX(inputs[i], wires=w)
            # Ansatz
            for r in range(config.reps):
                for w in range(config.num_qubits):
                    qml.Rot(params[r, w, 0], params[r, w, 1], params[r, w, 2], wires=w)
                # Entangling layer
                for w in range(config.num_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Return expectation values of Z on latent qubits
            return [qml.expval(qml.PauliZ(w)) for w in self.latent_wires]

        self.circuit = circuit

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Return the latent representation as expectation values."""
        return np.array(self.circuit(inputs, self.params))

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Linear decoder from latent to reconstructed data."""
        # Simple fixed decoder matrix for demonstration
        W = np.random.randn(self.config.latent_dim, len(self.data_wires))
        return latent @ W.T

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        latent = self.encode(inputs)
        return self.decode(latent)

    def loss(self, inputs: np.ndarray) -> float:
        recon = self.forward(inputs)
        return np.mean((recon - inputs) ** 2)

    def train(
        self,
        data: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> list[float]:
        """Parameter‑shift gradient descent training."""
        history: list[float] = []

        def loss_fn(params):
            old_params = self.params
            self.params = params
            loss_val = self.loss(data)
            self.params = old_params
            return loss_val

        for _ in range(epochs):
            grads = qml.grad(loss_fn)(self.params)
            self.params -= lr * grads
            history.append(self.loss(data))
        return history


__all__ = ["Autoencoder__gen406", "AutoencoderConfig"]
