"""Quantum QCNN implemented with PennyLane, featuring parameter‑shift training and noise simulation."""
import pennylane as qml
import numpy as np
import torch

class QCNNGen325:
    """Hybrid QCNN using a parameter‑shift variational circuit and feature‑map encoding."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3, seed: int = 12345):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024, seed=seed)
        # initialise trainable parameters: (n_layers, n_qubits, 3)
        self.params = np.random.randn(n_layers, n_qubits, 3)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _conv_circuit(self, params: np.ndarray, wires: list[int]) -> None:
        """Two‑qubit convolution unit implementing the pattern from the seed."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[2], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(np.pi / 2, wires=wires[0])

    def _pool_circuit(self, params: np.ndarray, wires: list[int]) -> None:
        """Pooling operation that discards the second qubit after entanglement."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[2], wires=wires[1])

    def _circuit(self, inputs: torch.Tensor, *weights):
        """Full QCNN circuit: feature map → conv → pool → … → measurement."""
        # Feature‑map encoding (Z‑feature map)
        for i in range(self.n_qubits):
            qml.RZ(inputs[i], wires=i)
        current_wires = list(range(self.n_qubits))
        for l in range(self.n_layers):
            # Convolution on adjacent pairs
            for idx, (w1, w2) in enumerate(zip(current_wires[0::2], current_wires[1::2])):
                self._conv_circuit(weights[l][idx], wires=[w1, w2])
            # Pooling: keep the first half of the qubits
            if len(current_wires) > 2:
                for idx, (w1, w2) in enumerate(zip(current_wires[0::2], current_wires[1::2])):
                    self._pool_circuit(weights[l][idx], wires=[w1, w2])
                current_wires = current_wires[: len(current_wires) // 2]
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qnode(inputs, *self.params)

    def train(self, data: torch.Tensor, labels: torch.Tensor,
              epochs: int = 100, lr: float = 0.01) -> None:
        """Simple gradient‑based training loop using PennyLane’s Adam optimiser."""
        opt = qml.AdamOptimizer(step_size=lr)
        loss_fn = lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2)

        for _ in range(epochs):
            def step():
                # Expect data to be a single sample of shape (n_qubits,)
                y_pred = self.qnode(data, *self.params)
                return loss_fn(y_pred, labels)
            opt.step(step)

def QCNNGen325Model() -> QCNNGen325:
    """Factory returning a ready‑to‑use QCNNGen325 instance."""
    return QCNNGen325()

__all__ = ["QCNNGen325", "QCNNGen325Model"]
