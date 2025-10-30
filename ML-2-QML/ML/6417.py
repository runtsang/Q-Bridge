"""Hybrid classical–quantum RBF kernel module with joint training.

The implementation keeps the original ``KernalAnsatz`` and ``Kernel`` classes
but augments them so that the quantum variant can be trained by using a
parameter‑free (in the sense that the quantum circuit is fixed) or
trainable circuit.  A new ``QuantumKernelMethod`` class provides a
trainable joint loss that compares the classical and quantum kernels.
The ``fit``/``predict`` API is compatible with scikit‑learn.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# Import the quantum kernel implementation
try:
    from QuantumKernelMethod_QML import Kernel as QuantumKernel
except Exception:
    # Minimal stub if the quantum module is unavailable
    class QuantumKernel(nn.Module):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("Quantum kernel not available")

        def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
            raise NotImplementedError("Quantum kernel not available")

# ----- Classical kernel classes (unchanged from the seed) -----
class KernalAnsatz(nn.Module):
    """Original RBF ansatz – kept for backward compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel that wraps the ``KernalAnsatz``."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def _kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    kernel: nn.Module,
) -> np.ndarray:
    """Compute a Gram matrix with the supplied kernel."""
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----- Hybrid kernel training -------------------------------------------------
class QuantumKernelMethod(nn.Module):
    """
    Hybrid classical–quantum kernel method.

    Parameters
    ----------
    gamma : float, default=1.0
        RBF gamma for the classical kernel.
    n_wires : int, default=4
        Number of qubits in the quantum kernel.
    lr : float, default=0.01
        Learning rate for the quantum parameters.
    epochs : int, default=200
        Number of optimisation steps.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        n_wires: int = 4,
        lr: float = 0.01,
        epochs: int = 200,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.n_wires = n_wires
        self.lr = lr
        self.epochs = epochs

        # Classical kernel (fixed)
        self.classical_kernel = Kernel(gamma=self.gamma)

        # Quantum kernel (trainable)
        self.quantum_kernel = QuantumKernel(n_wires=self.n_wires)

        # Optimiser for quantum parameters
        self.optimizer = Adam(self.quantum_kernel.parameters(), lr=self.lr)

        # Store training data
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ public API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelMethod":
        """
        Fit the quantum kernel to match the classical RBF kernel on the
        provided training data.

        The loss is the mean‑squared error between the two Gram matrices.
        """
        self.X_train = torch.from_numpy(X.astype(np.float32))
        self.y_train = torch.from_numpy(y.astype(np.float32))

        # Pre‑compute classical Gram matrix (treated as target)
        K_classical = _kernel_matrix(self.X_train, self.X_train, self.classical_kernel)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Quantum Gram matrix
            K_quantum = _kernel_matrix(self.X_train, self.X_train, self.quantum_kernel)

            loss = torch.mean((K_quantum - torch.from_numpy(K_classical)) ** 2)

            # Optional regularisation on the quantum parameters
            reg = 0.0
            for p in self.quantum_kernel.parameters():
                reg += torch.sum(p ** 2)
            loss += 1e-3 * reg

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"[epoch {epoch+1}/{self.epochs}] loss={loss.item():.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using a kernel ridge regression with the
        trained quantum kernel.  The ridge coefficient is fixed to
        ``alpha=1.0`` for simplicity.
        """
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_test = torch.from_numpy(X.astype(np.float32))
        K_train = _kernel_matrix(self.X_train, self.X_train, self.quantum_kernel)
        K_test = _kernel_matrix(X_test, self.X_train, self.quantum_kernel)

        # Kernel ridge regression: alpha = 1.0
        alpha = 1.0
        coef = torch.linalg.solve(K_train + alpha * torch.eye(K_train.shape[0]), self.y_train)
        y_pred = K_test @ coef
        return y_pred.detach().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean squared error of the predictions."""
        y_pred = self.predict(X)
        return float(np.mean((y_pred - y) ** 2))

    # ----------------------------------------------------------------- utilities
    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the Gram matrix using the trained quantum kernel."""
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        a_t = torch.from_numpy(a.astype(np.float32))
        b_t = torch.from_numpy(b.astype(np.float32))
        return _kernel_matrix(a_t, b_t, self.quantum_kernel)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "_kernel_matrix",
    "QuantumKernelMethod",
]
