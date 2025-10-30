"""Variational quantum fully‑connected layer.

This module implements a parameterised quantum circuit that mimics a
classical fully‑connected layer.  The circuit consists of a chain of
single‑qubit Ry rotations that encode the input features, followed by a
trainable layer of Ry gates and a simple CNOT entanglement pattern.
The expectation value of the Pauli‑Z operator on the first qubit is
returned as a NumPy array.

The class also offers a ``train`` helper that optimises the circuit
parameters with respect to a mean‑squared‑error loss using the
autograd engine of Pennylane.

Typical usage:

>>> qmodel = FCL(n_qubits=4)
>>> preds = qmodel.run([0.1, 0.2, 0.3, 0.4])
>>> qmodel.train(X, y, epochs=5, lr=0.01)
"""

import pennylane as qml
import pennylane.numpy as pnp
import numpy as np


class FCL:
    """Parameterised quantum circuit acting as a fully‑connected layer."""

    def __init__(self, n_qubits: int = 1, entanglement: str = "cnot") -> None:
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Trainable parameters
        self.theta = pnp.random.randn(n_qubits, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, theta):
            # Feature encoding
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Trainable rotations
            for i in range(self.n_qubits):
                qml.RY(theta[i], wires=i)
            # Entanglement
            if self.entanglement == "cnot":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: list[float] | np.ndarray) -> np.ndarray:
        """Compute the expectation value for the supplied input vector."""
        x = np.array(thetas, dtype=np.float64)
        if x.shape[0]!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {x.shape[0]}"
            )
        val = self.circuit(x, self.theta)
        return np.array([val])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 10,
        lr: float = 0.01,
    ) -> None:
        """Train the circuit parameters on a regression task.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_qubits)
            Input angles for each sample.
        y : ndarray of shape (n_samples, 1)
            Target values.
        epochs : int, default 10
            Number of optimisation epochs.
        lr : float, default 0.01
            Learning rate.
        """
        opt = qml.GradientDescentOptimizer(lr)

        for _ in range(epochs):
            def loss_fn(theta):
                preds = np.array([self.circuit(x, theta) for x in X])
                return np.mean((preds - y.squeeze()) ** 2)

            self.theta = opt.step(loss_fn, self.theta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience wrapper that uses the trained parameters."""
        if self.theta is None:
            raise RuntimeError("Model has not been trained yet.")
        preds = np.array([self.circuit(x, self.theta) for x in X])
        return preds.reshape(-1, 1)


__all__ = ["FCL"]
