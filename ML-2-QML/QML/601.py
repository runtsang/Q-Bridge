"""Quantum kernel construction using a trainable variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["QuantumKernelMethod"]


class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel based on a variational circuit with trainable Ry weights."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Trainable weights for Ry gates
        self.weights = nn.Parameter(torch.ones(self.n_wires))
        # Define ansatz list
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode inputs x and y and compute overlap."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for gate in self.ansatz:
            params = x[:, gate["input_idx"][0]] * self.weights[gate["input_idx"][0]]
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)
        # Encode y with negative params
        for gate in reversed(self.ansatz):
            params = -y[:, gate["input_idx"][0]] * self.weights[gate["input_idx"][0]]
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
        """Compute the Gram matrix between X and Y."""
        K = torch.zeros((X.shape[0], Y.shape[0]))
        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                self.forward(self.q_device, xi.unsqueeze(0), yj.unsqueeze(0))
                K[i, j] = torch.abs(self.q_device.states.view(-1)[0]).item()
        return K.numpy()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.01) -> None:
        """Train the variational weights to improve classification."""
        optimizer = optim.Adam([self.weights], lr=lr)
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.long)
        for epoch in range(epochs):
            optimizer.zero_grad()
            K = self.kernel_matrix(X_torch, X_torch)
            # Simple hinge loss: maximize similarity for same class, minimize for different
            loss = 0.0
            for i in range(len(y)):
                for j in range(len(y)):
                    if y[i] == y[j]:
                        loss -= K[i, j]
                    else:
                        loss += K[i, j]
            loss = loss / (len(y) ** 2)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Predict class labels using nearest neighbor in kernel space."""
        K_test = self.kernel_matrix(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(X_train, dtype=torch.float32))
        idx = torch.argmax(torch.tensor(K_test), dim=1)
        return y_train[idx].numpy()
