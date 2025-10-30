"""Quantum kernel module using PennyLane with a trainable variational ansatz."""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Optional

__all__ = ["QuantumKernelMethod", "QuantumKernelMethodConfig"]

class QuantumKernelMethodConfig:
    """Configuration for the PennyLane quantum kernel."""
    def __init__(
        self,
        n_qubits: int = 4,
        device_name: str = "default.qubit",
        device_kwargs: dict | None = None,
        lr: float = 0.01,
        epochs: int = 200,
    ):
        self.n_qubits = n_qubits
        self.device_name = device_name
        self.device_kwargs = device_kwargs or {}
        self.lr = lr
        self.epochs = epochs

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel implemented with PennyLane, providing a trainable quantum ansatz."""

    def __init__(self, config: QuantumKernelMethodConfig):
        super().__init__()
        self.config = config
        self.device = qml.device(config.device_name, wires=config.n_qubits, **config.device_kwargs)
        # Parameters: one rotation (3 angles) per qubit
        self.params = nn.Parameter(torch.randn(config.n_qubits, 3))
        self.n_qubits = config.n_qubits
        self.svm: Optional[SVC] = None

    def _encode(self, x: torch.Tensor) -> None:
        """Encode classical data via Ry rotations."""
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)

    def _variational_layer(self) -> None:
        """Variational layer with parametrised rotations and CNOTs."""
        for i in range(self.n_qubits):
            qml.Rot(*self.params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def _kernel_qnode(self, x: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.device, interface="torch")
        def circuit(x):
            self._encode(x)
            self._variational_layer()
            return qml.state()
        return circuit(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the absolute overlap between two encoded states."""
        state_x = self._kernel_qnode(x)
        state_y = self._kernel_qnode(y)
        return torch.abs(torch.sum(state_x.conj() * state_y, dim=-1))

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix for two datasets."""
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        n, m = X_t.shape[0], Y_t.shape[0]
        K = torch.empty((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self.forward(X_t[i], Y_t[j])
        return K.detach().cpu().numpy()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the variational parameters using a contrastive loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        for epoch in range(self.config.epochs):
            loss = 0.0
            for i in range(len(X_t)):
                xi = X_t[i].unsqueeze(0)
                for j in range(i + 1, len(X_t)):
                    xj = X_t[j].unsqueeze(0)
                    label = (y_t[i] == y_t[j]).float()
                    k = self.forward(xi, xj)
                    loss += label * (1 - k) ** 2 + (1 - label) * k ** 2
            loss = loss / (len(X_t) * (len(X_t) - 1) / 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 50 == 0:
                print(f"[PennyLane] epoch {epoch} loss {loss.item():.4f}")

    def fit(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> None:
        """Fit an SVM with the pre‑computed kernel."""
        K = self.kernel_matrix(X, X)
        self.svm = SVC(C=C, kernel="precomputed")
        self.svm.fit(K, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new samples."""
        if self.svm is None:
            raise RuntimeError("Model has not been fitted.")
        K = self.kernel_matrix(X, self.svm.support_vectors_)
        return self.svm.predict(K)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
