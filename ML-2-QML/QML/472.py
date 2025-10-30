"""Hybrid variational quantum neural network for regression.

Key features
------------
* Parameterised ansatz with entanglement across all qubits.
* Multiple Pauli‑Z measurements for richer observables.
* Automatic differentiation via Pennylane’s autograd interface.
* Simple gradient‑descent training loop.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np


class EstimatorQNN:
    """
    Hybrid QNN that maps a vector of input features to a regression output.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits for the variational circuit.
    n_layers : int, default 2
        Depth of the ansatz.
    device : str, default "default.qubit"
        PennyLane device name.
    use_variance : bool, default False
        If True, returns a vector of Pauli‑Z expectation values (per qubit) as a
        variance estimate; otherwise returns the mean of all Z measurements.
    """
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = "default.qubit",
        use_variance: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_variance = use_variance
        self.dev = qml.device(device, wires=n_qubits)

        # Trainable rotation parameters: shape (layers, qubits, 3)
        self.weight_params = np.random.randn(n_layers, n_qubits, 3)

        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, *input_features, weight_params):
        """Variational circuit with input encoding and entanglement."""
        # Input encoding: rotate each qubit by the corresponding feature
        for i, w in enumerate(input_features):
            qml.RY(w, wires=i)

        # Ansatz layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.Rot(*weight_params[layer, qubit], wires=qubit)
            # All‑to‑all entanglement (chain + wrap‑around)
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

        # Measurements
        if self.use_variance:
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        else:
            return sum([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]) / self.n_qubits

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        inputs : array, shape (batch, n_qubits)
            Input feature batch.

        Returns
        -------
        preds : array
            Predictions of shape (batch,) or (batch, n_qubits) if use_variance.
        """
        preds = []
        for x in inputs:
            preds.append(self.qnode(*x, weight_params=self.weight_params))
        return np.array(preds)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 200,
    ) -> None:
        """
        Simple gradient‑descent training loop.

        Parameters
        ----------
        X : array, shape (N, n_qubits)
            Training inputs.
        y : array, shape (N,) or (N, n_qubits)
            Target outputs.
        lr : float, default 0.01
            Learning rate.
        epochs : int, default 200
            Number of epochs.
        """
        opt = qml.GradientDescentOptimizer(lr)

        def loss_fn(params):
            preds = np.array([self.qnode(*x, weight_params=params) for x in X])
            return np.mean((preds - y) ** 2)

        for epoch in range(epochs):
            self.weight_params = opt.step(loss_fn, self.weight_params)
            if epoch % 20 == 0:
                loss = loss_fn(self.weight_params)
                print(f"Epoch {epoch:3d} – loss: {loss:.4f}")

__all__ = ["EstimatorQNN"]
