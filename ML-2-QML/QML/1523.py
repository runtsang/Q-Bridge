"""Hybrid quantum kernel module using PennyLane variational ansatz.

The kernel is defined as the absolute overlap between two encoded states.
The variational parameters are trained to maximise the kernel for similar
data points while minimising it for dissimilar points.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence, Callable, Optional


class Kernel(torch.nn.Module):
    """Quantum kernel implemented with a trainable PennyLane ansatz.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits for the device. Default is 4.
    depth : int, optional
        Number of layers in the variational circuit. Default is 2.
    seed : int, optional
        Random seed for weight initialization.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2, seed: int | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.seed = seed
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        # Initialise trainable parameters
        rng = np.random.default_rng(seed)
        self.params = torch.tensor(
            rng.normal(size=(self.depth, self.n_wires, 3)), dtype=torch.float64, requires_grad=True
        )

    def _encode(self, x: torch.Tensor) -> None:
        """Encode classical data into quantum rotations."""
        for i, wire in enumerate(range(self.n_wires)):
            qml.RX(x[0, i], wires=wire)
            qml.RY(x[0, i] * 0.5, wires=wire)

    def _variate(self, params: torch.Tensor) -> None:
        """Variational layer."""
        for layer in range(self.depth):
            for wire in range(self.n_wires):
                qml.Rot(
                    params[layer, wire, 0],
                    params[layer, wire, 1],
                    params[layer, wire, 2],
                    wires=wire,
                )
            # Entangling layer
            for wire in range(self.n_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value between two samples."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(x_vec: torch.Tensor, y_vec: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            self._encode(x_vec)
            self._variate(params)
            state1 = qml.state()

            qml.reset()
            self._encode(y_vec)
            self._variate(params)
            state2 = qml.state()

            return torch.abs(torch.sum(state1.conj() * state2))

        return circuit(x, y, self.params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper that ensures correct shapes."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return self._kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences of tensors."""
        a = torch.stack(a)
        b = torch.stack(b)
        mat = torch.zeros((len(a), len(b)), dtype=torch.float64)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.forward(xi, yj)
        return mat.detach().cpu().numpy()

    def train(self, X: torch.Tensor, labels: torch.Tensor, lr: float = 0.02, epochs: int = 300) -> None:
        """Train the variational parameters to discriminate class labels.

        Loss is mean squared error between kernel diagonal (self‑similarity) and 1 for same
        class, 0 for different class.
        """
        optimizer = torch.optim.Adam([self.params], lr=lr)
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            K = self.kernel_matrix([X], [X])  # compute full matrix
            K_tensor = torch.tensor(K, dtype=torch.float64)
            same = (labels[:, None] == labels[None, :]).float()
            loss = torch.mean((K_tensor - same) ** 2)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch} – loss {loss.item():.4f}")


__all__ = ["Kernel"]
