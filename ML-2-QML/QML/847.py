"""Quantum kernel construction using a variational circuit with trainable rotation angles."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import pennylane as qml

class QuantumKernelMethod(nn.Module):
    """
    Quantum kernel built from a parameterised circuit. The rotation angles are trained
    end‑to‑end to minimise a regression loss. The kernel is the squared overlap between
    the states prepared for two data points.
    """

    def __init__(self,
                 n_wires: int = 4,
                 n_layers: int = 2,
                 lambda_reg: float = 1e-5,
                 lr: float = 0.01,
                 epochs: int = 200):
        """
        Parameters
        ----------
        n_wires : int
            Number of qubits.
        n_layers : int
            Depth of the variational circuit.
        lambda_reg : float
            Regularisation strength for the ridge regression.
        lr : float
            Learning rate for the optimiser.
        epochs : int
            Number of optimisation iterations.
        """
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs

        # Trainable parameters: rotation angles for each layer and wire
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, dtype=torch.float32))

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_wires)

        # QNode that prepares a state and returns the state vector
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, params: torch.Tensor):
            # Encode classical data into qubit rotations
            for i in range(n_wires):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_wires):
                    qml.RZ(params[layer, i], wires=i)
                # Entangling layer (full‑chain)
                for i in range(n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_wires - 1, 0])
            return qml.state()

        self.circuit = circuit

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix using the quantum kernel.
        """
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        # Evaluate the state for each sample
        states_X = torch.stack([self.circuit(x, self.params) for x in X])  # (n, 2^n)
        states_Y = torch.stack([self.circuit(y, self.params) for y in Y])  # (m, 2^n)

        # Squared overlap: |<ψ(x)|ψ(y)>|^2
        return torch.abs(torch.matmul(states_X, states_Y.conj().t())) ** 2

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the quantum kernel matrix.
        """
        return self.kernel_matrix(X, Y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelMethod":
        """
        Fit the kernel ridge regression model.
        Optimises the circuit parameters and computes the regression coefficients.
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.X_train = X
        self.y_train = y

        optimizer = torch.optim.Adam([self.params], lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            K = self.kernel_matrix(X, X)
            K_reg = K + self.lambda_reg * torch.eye(K.shape[0], device=K.device)
            alpha = torch.linalg.solve(K_reg, y)
            y_pred = K @ alpha
            loss = torch.mean((y_pred - y) ** 2)
            loss.backward()
            optimizer.step()

        # Store final regression coefficients
        K = self.kernel_matrix(X, X)
        K_reg = K + self.lambda_reg * torch.eye(K.shape[0], device=K.device)
        self.alpha = torch.linalg.solve(K_reg, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        """
        X = torch.tensor(X, dtype=torch.float32)
        K_test = self.kernel_matrix(X, self.X_train)
        return (K_test @ self.alpha).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
