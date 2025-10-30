"""Quantum implementation of an autoencoder using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode


@dataclass
class QAutoencoderConfig:
    """Parameters for the quantum autoencoder."""
    num_qubits: int
    latent_dim: int
    reps: int = 3
    shots: int = 1024


class MultiModalAutoencoder:
    """A variational quantum autoencoder with angle encoding and a full‑reversal ansatz."""
    def __init__(self, cfg: QAutoencoderConfig) -> None:
        self.cfg = cfg
        self.dev = qml.device("default.qubit", wires=cfg.num_qubits, shots=cfg.shots)
        # Initialise variational parameters
        self.params = np.random.uniform(0, 2 * np.pi, (cfg.reps, cfg.num_qubits))
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Full circuit that encodes, variationally processes, and measures."""
        # Angle embedding of the input into the first `latent_dim` qubits
        for i in range(self.cfg.latent_dim):
            qml.RX(x[i], wires=i)
        # Prepare remaining qubits in |+> to serve as trash
        for i in range(self.cfg.latent_dim, self.cfg.num_qubits):
            qml.Hadamard(wires=i)
        # Variational ansatz
        for r in range(self.cfg.reps):
            for w in range(self.cfg.num_qubits):
                qml.Rot(params[r, w], 0, 0, wires=w)
            # Entangling layer (cyclic CNOT chain)
            for w in range(self.cfg.num_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[self.cfg.num_qubits - 1, 0])
        # Measure all qubits in the Pauli‑Z basis
        return qml.expval(qml.PauliZ(wires=range(self.cfg.num_qubits)))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode classical input to a quantum‑derived latent vector."""
        return self.qnode(x, self.params)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode quantum latent back to classical space (identity decoder)."""
        return self.encode(z)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full autoencoder pass."""
        return self.decode(self.encode(x))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean squared reconstruction error."""
        return np.mean((x - y) ** 2)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> List[float]:
        """Batch‑level training loop with a simple gradient‑descent optimiser."""
        opt = qml.GradientDescentOptimizer(lr)
        history: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x in data:
                loss, grads = opt.step_and_grad(lambda p: self.loss(x, self.forward(x)), self.params)
                self.params = opt.apply_gradients(grads, self.params)
                epoch_loss += loss
            epoch_loss /= len(data)
            history.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1:3d}/{epochs}  loss: {epoch_loss:.4f}")
        return history


__all__ = ["MultiModalAutoencoder", "QAutoencoderConfig"]
