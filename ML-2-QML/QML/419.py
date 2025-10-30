"""Quantum neural network regressor using Pennylane.

A 2‑qubit variational circuit with trainable rotation angles and
entangling gates. The network outputs the expectation value of
Pauli‑Z on the first qubit, which is used as the regression score.
"""
import pennylane as qml
import numpy as np

class EstimatorQNN:
    """A variational quantum circuit wrapped as a callable regressor.

    Parameters
    ----------
    wires : int
        Number of qubits (default 2).
    layers : int
        Number of variational layers.
    """
    def __init__(self, wires: int = 2, layers: int = 2):
        self.wires = wires
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=wires)
        # Rotation parameters: shape (layers, wires, 3) for RX, RY, RZ
        self.params = np.random.randn(layers, wires, 3)
        self._build_qnode()

    def _rotation_layer(self, params):
        """Apply RX, RY, RZ rotations on each qubit."""
        for i, (rx, ry, rz) in enumerate(params):
            qml.RX(rx, wires=i)
            qml.RY(ry, wires=i)
            qml.RZ(rz, wires=i)

    def _entangle(self):
        """Entangle qubits with a chain of CNOTs."""
        for i in range(self.wires - 1):
            qml.CNOT(wires=[i, i + 1])

    def _qnode(self, x, weights):
        """Quantum node computing expectation of Pauli‑Z on qubit 0."""
        # Simple input encoding: RY rotations per qubit
        for i in range(self.wires):
            qml.RY(x[i], wires=i)

        # Variational layers
        for layer in range(self.layers):
            self._rotation_layer(weights[layer])
            self._entangle()

        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the circuit for a single input vector."""
        x = np.clip(x, -1.0, 1.0)  # simple input scaling
        return float(self._qnode(x, self.params))

    @classmethod
    def train(cls,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 200,
              lr: float = 0.01) -> "EstimatorQNN":
        """Train the variational parameters using gradient descent."""
        model = cls()
        opt = qml.GradientDescentOptimizer(stepsize=lr)

        for _ in range(epochs):
            for xi, yi in zip(X, y):
                def loss_fn(w):
                    pred = model._qnode(xi, w)
                    return (pred - yi) ** 2

                model.params = opt.step(loss_fn, model.params)
        return model

__all__ = ["EstimatorQNN"]
