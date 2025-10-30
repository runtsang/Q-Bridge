"""Quantum autoencoder using Pennylane with amplitude encoding and hybrid training."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Tuple, Callable, List, Optional
from dataclasses import dataclass
from functools import partial


@dataclass
class QuantumAutoencoderGen049Config:
    """Configuration for the quantum autoencoder."""
    num_qubits: int
    latent_dim: int
    num_layers: int = 3
    reps: int = 2
    optimization_steps: int = 200
    lr: float = 0.01
    seed: int = 42


class QuantumAutoencoderGen049:
    """Variational quantum autoencoder built on Pennylane."""

    def __init__(self, config: QuantumAutoencoderGen049Config) -> None:
        self.config = config
        self.dev = qml.device("default.qubit", wires=self.config.num_qubits)
        self._build_ansatz()

    def _build_ansatz(self) -> None:
        """Constructs a parameterized ansatz with amplitude encoding."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Amplitude encode the input
            qml.AmplitudeEmbedding(
                features=x,
                wires=range(self.config.latent_dim),
                normalize=True,
            )
            # Variational layers
            for layer in range(self.config.num_layers):
                qml.templates.StronglyEntanglingLayers(params[layer], wires=range(self.config.num_qubits))
            # Measure latent qubits
            return qml.expval(qml.PauliZ(0))  # example measurement; extend as needed

        self.circuit = circuit
        # Initialize parameters
        self.params = pnp.random.uniform(
            low=-np.pi, high=np.pi,
            size=(self.config.num_layers, self.config.num_qubits, 3)
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return latent representation as expectation values."""
        return self.circuit(x, self.params)

    def loss(self, x: np.ndarray) -> float:
        """MSE between input and decoded output."""
        # Simple autoencoder: decode by re‑amplitude‑encoding and measuring
        latent = self.encode(x)
        # For demonstration, we treat latent as reconstruction (identity)
        return np.mean((x - latent) ** 2)

    def train(self, data: np.ndarray, *, verbose: bool = True) -> List[float]:
        """Hybrid training loop using gradient descent."""
        opt = qml.GradientDescentOptimizer(self.config.lr)
        history: List[float] = []
        params = self.params

        for step in range(self.config.optimization_steps):
            params, loss_val = opt.step_and_cost(self.loss, params, data)
            self.params = params
            history.append(loss_val)
            if verbose and (step + 1) % 20 == 0:
                print(f"Step {step+1}/{self.config.optimization_steps} – loss: {loss_val:.6f}")

        return history

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Return the latent vector for a single data point."""
        return self.encode(x)

    def save(self, path: str) -> None:
        """Persist parameters."""
        np.save(path, self.params)

    @classmethod
    def load(cls, path: str, config: QuantumAutoencoderGen049Config) -> "QuantumAutoencoderGen049":
        """Load parameters from disk."""
        instance = cls(config)
        instance.params = np.load(path)
        return instance

    def __repr__(self) -> str:
        return f"<QuantumAutoencoderGen049 num_qubits={self.config.num_qubits} latent_dim={self.config.latent_dim}>"


__all__ = ["QuantumAutoencoderGen049", "QuantumAutoencoderGen049Config"]
