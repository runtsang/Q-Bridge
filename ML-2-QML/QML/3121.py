"""Quantum autoencoder using Pennylane: 2×2 patch → 4‑qubit features."""

import pennylane as qml
import numpy as np

class QuanvolutionAutoencoder:
    """Quantum autoencoder: encodes 2×2 image patches using a variational circuit."""
    def __init__(self, num_qubits: int = 4, wires: list[int] | None = None):
        self.num_qubits = num_qubits
        self.wires = wires or list(range(num_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, patch: np.ndarray, weights: np.ndarray):
        # patch shape (4,) with pixel values in [0, 1]
        for i, w in enumerate(patch):
            qml.RY(w, wires=self.wires[i])
        # Variational layer
        qml.layer(qml.StronglyEntanglingLayers, 2, wires=self.wires)(weights)
        # Domain wall: X on qubits 2..num_qubits-1
        for i in range(2, self.num_qubits):
            qml.PauliX(wires=self.wires[i])
        # Measurement: expectation of PauliZ on all qubits
        return [qml.expval(qml.PauliZ(self.wires[i])) for i in range(self.num_qubits)]

    def forward(self, patches: np.ndarray) -> np.ndarray:
        """Compute quantum features for a batch of 2×2 patches."""
        batch_size = patches.shape[0]
        # Random weights for demonstration; in practice these would be trainable
        weights = np.random.randn(2, self.num_qubits, 3)
        out = np.zeros((batch_size, self.num_qubits))
        for i in range(batch_size):
            out[i] = self.qnode(patches[i], weights)
        return out

__all__ = ["QuanvolutionAutoencoder"]
