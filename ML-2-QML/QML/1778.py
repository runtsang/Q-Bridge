"""
Quantum QCNN implemented with Pennylane.
The architecture follows the same high‑level convolution‑pooling pattern as the
original seed while providing a parameter‑shift optimiser and a simple
feature‑embedding using AngleEmbedding.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer


class QCNNHybrid:
    """
    Quantum convolutional neural network with a lightweight variational ansatz.
    The network is built on a default.qubit simulator and returns the expectation
    value of a single Pauli‑Z observable after feature embedding.
    """

    def __init__(self,
                 n_qubits: int = 8,
                 n_layers: int = 3,
                 seed: int | None = 42) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used in the circuit.
        n_layers : int
            Number of convolution‑pooling blocks.
        seed : int | None
            Random seed for weight initialization.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        rng = np.random.default_rng(seed)
        # Each layer has n_qubits/2 convolution weights, each with 3 parameters
        self.params = rng.normal(size=(n_layers, n_qubits // 2, 3))

        # Observable for the final measurement
        self.observable = qml.PauliZ(0)

    def _conv_layer(self, layer: int, wires: np.ndarray) -> None:
        """
        Convolutional block that couples pairs of qubits with a small ansatz.
        """
        for i in range(0, len(wires), 2):
            idx = i // 2
            w = self.params[layer, idx]
            qml.RZ(w[0], wires=w[i])
            qml.RY(w[1], wires=w[i + 1])
            qml.CNOT(wires=[w[i], w[i + 1]])
            qml.RZ(w[2], wires=w[i + 1])

    def _pool_layer(self, wires: np.ndarray) -> None:
        """
        Simple pooling that measures and discards one qubit per pair.
        Here implemented as a measurement‑induced collapse via a
        partial trace (simulated by a basis measurement and re‑initialisation).
        """
        for i in range(0, len(wires), 2):
            # Measure the second qubit of the pair
            qml.Measure(wires=wires[i + 1])
            # Reset it to |0⟩ (simulated by a reset operation)
            qml.Reset(wires=wires[i + 1])

    @qml.qnode
    def circuit(self, x: np.ndarray, params: np.ndarray) -> float:
        """
        QNode that executes the full QCNN and returns the expectation value.
        """
        # Feature embedding
        qml.templates.AngleEmbedding(x, wires=range(self.n_qubits))
        for layer in range(self.n_layers):
            self._conv_layer(layer, range(self.n_qubits))
            self._pool_layer(range(self.n_qubits))
        return qml.expval(self.observable)

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: evaluate the circuit with the current parameters.
        """
        return self.circuit(x, self.params)

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 200,
              lr: float = 0.01) -> None:
        """
        Simple training loop using Adam optimiser and a binary cross‑entropy loss.
        """
        opt = AdamOptimizer(lr)
        for epoch in range(epochs):
            loss = 0.0
            for xi, yi in zip(X, y):
                pred = self.forward(xi)
                # Binary cross‑entropy loss
                loss += -yi * np.log(pred + 1e-10) - (1 - yi) * np.log(1 - pred + 1e-10)
            loss /= len(X)
            grads = qml.gradients.param_shift(self.circuit)(X[0], self.params)
            self.params -= lr * grads
            if epoch % 20 == 0:
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

__all__ = ["QCNNHybrid"]
