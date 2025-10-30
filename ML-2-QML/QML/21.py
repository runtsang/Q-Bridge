"""Quantum kernel implementation using Pennylane."""

import pennylane as qml
import numpy as np
import torch
from torch import nn
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Parameterized quantum kernel ansatz.

    The circuit encodes two classical data vectors `x` and `y` into a single
    quantum state via Ry rotations followed by a fixed entangling layer.
    The kernel value is the absolute square of the overlap between the
    two states, i.e., |<ψ(x)|ψ(y)>|^2.
    """
    def __init__(self, n_wires: int = 4, entangler_depth: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.entangler_depth = entangler_depth
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(self.dev, interface="torch")
        def _qnode(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode first vector
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Entangling layer
            for _ in range(self.entangler_depth):
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Encode second vector with negative sign
            for i in range(self.n_wires):
                qml.RY(-y[i], wires=i)
            return qml.state()
        self._qnode = _qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value for two 1‑D tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of length equal to `n_wires`.

        Returns
        -------
        torch.Tensor
            Kernel value (scalar) as a 1‑D tensor.
        """
        state = self._qnode(x, y)
        return torch.abs(state[0]) ** 2

class Kernel(nn.Module):
    """Convenience wrapper that returns a kernel matrix."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.ansatz = KernalAnsatz(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between two batches of data.

        Parameters
        ----------
        x, y : torch.Tensor
            Tensors of shape (batch, n_wires).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_x, batch_y).
        """
        n_x = x.shape[0]
        n_y = y.shape[0]
        kernels = torch.empty((n_x, n_y), dtype=torch.float32)
        for i in range(n_x):
            for j in range(n_y):
                kernels[i, j] = self.ansatz(x[i], y[j])
        return kernels

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of 1‑D tensors."""
    kernel = Kernel(n_wires=a[0].shape[0])
    a_flat = [x.reshape(-1) for x in a]
    b_flat = [y.reshape(-1) for y in b]
    return np.array([[kernel(x, y).item() for y in b_flat] for x in a_flat])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]

# Test harness
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    Y = torch.randn(4, 4)
    k = Kernel(n_wires=4)
    print("Quantum kernel matrix:\\n", k(X, Y).detach().numpy())
