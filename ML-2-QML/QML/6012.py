"""Quantum kernel construction using PennyLane.

This module extends the original quantum kernel implementation by
* adding a multi‑layer parameterized ansatz,
* supporting batched computation of the Gram matrix,
* providing a helper to train a linear SVM on the quantum kernel.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence
from sklearn.svm import LinearSVC

class KernalAnsatz:
    """Quantum data‑encoding ansatz with configurable layers."""
    def __init__(self, n_wires: int, n_layers: int = 1) -> None:
        self.n_wires = n_wires
        self.n_layers = n_layers

    def encode(self, x: torch.Tensor, y: torch.Tensor = None) -> None:
        """Encode data vector x and optionally y with negative sign."""
        for _ in range(self.n_layers):
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
        if y is not None:
            for _ in range(self.n_layers):
                for i in range(self.n_wires):
                    qml.RY(-y[i], wires=i)
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])

class Kernel:
    """Quantum kernel evaluated via a fixed PennyLane ansatz."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires, self.n_layers)

    def _circuit(self, x: torch.Tensor, y: torch.Tensor) -> qml.QNode:
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            self.ansatz.encode(x, y)
            return qml.state()
        return circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel between two single‑sample vectors.
        Supports batched inputs by looping over the batch dimension.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            circuit = self._circuit(x[i], y[i])
            state = circuit()
            outputs.append(torch.abs(state[0]) ** 2)
        return torch.stack(outputs)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between two lists of tensors.
    Returns a 2‑D numpy array of shape (len(a), len(b)).
    """
    kernel = Kernel()
    a_t = torch.stack(a)
    b_t = torch.stack(b)
    mat = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, x in enumerate(a_t):
        for j, y in enumerate(b_t):
            mat[i, j] = kernel(x, y).item()
    return mat

def train_svm(gram: np.ndarray, y: np.ndarray, **svm_kwargs) -> LinearSVC:
    """Train a linear SVM directly on the provided Gram matrix."""
    clf = LinearSVC(**svm_kwargs)
    clf.fit(gram, y)
    return clf

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "train_svm"]
