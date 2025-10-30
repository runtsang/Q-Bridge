"""Quantum kernel using a parameter‑shift ansatz implemented with Pennylane."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml

class ParameterShiftAnsatz:
    """Simple parameter‑shift ansatz that encodes data into a 2‑qubit circuit."""
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def encode(self, data: np.ndarray) -> None:
        """Store data for later use in the circuit."""
        self.data = data

    def circuit(self, params: np.ndarray) -> None:
        """Quantum circuit that applies Ry rotations with data‑dependent parameters."""
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        qml.CNOT(wires=[0, 1])

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the overlap between two encoded states."""
        self.encode(x)
        @qml.qnode(self.dev)
        def state_x():
            self.circuit(self.data)
            return qml.state()
        psi_x = state_x()
        self.encode(y)
        @qml.qnode(self.dev)
        def state_y():
            self.circuit(self.data)
            return qml.state()
        psi_y = state_y()
        return np.abs(np.vdot(psi_x, psi_y)) ** 2

class QuantumKernelMethod:
    """Wrapper that provides a quantum kernel and a simple kernel ridge regression."""
    def __init__(self, n_qubits: int = 2, alpha: float = 1.0) -> None:
        self.ansatz = ParameterShiftAnsatz(n_qubits)
        self.alpha = alpha
        self.train_X = None
        self.train_y = None
        self.train_K_inv = None

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of data."""
        n, m = X.shape[0], Y.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self.ansatz.kernel(X[i], Y[j])
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit kernel ridge regression."""
        K = self.kernel_matrix(X, X)
        n = K.shape[0]
        K_reg = K + self.alpha * np.eye(n)
        self.train_K_inv = np.linalg.inv(K_reg)
        self.train_X = X
        self.train_y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        if self.train_K_inv is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_test = self.kernel_matrix(X, self.train_X)
        return K_test @ self.train_K_inv @ self.train_y

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """Compute kernel matrix between two lists of numpy arrays."""
    model = QuantumKernelMethod()
    X = np.vstack(a)
    Y = np.vstack(b)
    return model.kernel_matrix(X, Y)

__all__ = ["ParameterShiftAnsatz", "QuantumKernelMethod", "kernel_matrix"]
