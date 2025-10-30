"""Quantum neural network regressor using PennyLane.

The EstimatorQNN class implements a variational circuit with a
parameter‑shaped encoding of the input and trainable rotation gates.
It can be trained with a classical optimiser and used for regression.
It is fully compatible with scikit‑learn pipelines.
"""
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

class EstimatorQNN(BaseEstimator, RegressorMixin):
    """Quantum neural network regressor."""
    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        lr: float = 0.01,
        epochs: int = 200,
        device: str = "default.qubit",
        verbose: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.device_name = device
        self.verbose = verbose
        self._weights: torch.Tensor | None = None

    def _build_qnode(self):
        dev = qml.device(self.device_name, wires=self.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor, w: torch.Tensor):
            # Input encoding
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(w[layer, i], wires=i)
                # Entangling CNOT chain
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliY(0))

        return circuit

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self._weights = torch.randn(self.n_layers, self.n_qubits, requires_grad=True)

        circuit = self._build_qnode()
        optimizer = optim.Adam([self._weights], lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = torch.stack([circuit(x, self._weights) for x in X_t]).unsqueeze(1)
            loss = loss_fn(preds, y_t)
            loss.backward()
            optimizer.step()
            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d} loss: {loss.item():.6f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        circuit = self._build_qnode()
        preds = torch.stack([circuit(torch.tensor(x, dtype=torch.float32), self._weights) for x in X])
        return preds.detach().numpy().flatten()

__all__ = ["EstimatorQNN"]
