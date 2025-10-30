"""Hybrid classical‑quantum kernel implemented with PennyLane.

The :class:`QuantumKernelMethod` class offers the same public interface as
the TorchQuantum version but uses a variational circuit to compute the
quantum kernel.  It also provides a lightweight grid‑search tuner for
the RBF bandwidth and the quantum weight.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn
import pennylane as qml

# --------------------------------------------------------------------------- #
# 1.  Classical RBF kernel
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial Basis Function (RBF) kernel."""
    def __init__(self, gamma: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=-1)
        k = torch.exp(-self.gamma * sq_norm)
        if self.normalize:
            k = k / torch.sqrt(k.diagonal(dim1=-2, dim2=-1).unsqueeze(1))
        return k

# --------------------------------------------------------------------------- #
# 2.  Quantum kernel (PennyLane)
# --------------------------------------------------------------------------- #
class QuantumAnsatz(nn.Module):
    """Hardware‑efficient ansatz that encodes two vectors onto the same qubits."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=self.n_wires)

    def _circuit(self, params: np.ndarray) -> np.ndarray:
        """Return the overlap of two encoded states."""
        # Encode first vector
        for i, w in enumerate(range(self.n_wires)):
            qml.RY(params[i], wires=w)
        # Entanglement layer
        for w in range(self.n_wires - 1):
            qml.CNOT(wires=[w, w + 1])
        # Undo encoding of second vector with negative params
        for i, w in enumerate(range(self.n_wires)):
            qml.RY(-params[i], wires=w)
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        K = torch.empty((N, N))
        for i in range(N):
            for j in range(N):
                params = torch.cat([x[i], y[j]])
                val = self._circuit(params.detach().cpu().numpy())
                K[i, j] = torch.tensor(val, dtype=torch.float32)
        return K

class QuantumKernel(nn.Module):
    """Wraps :class:`QuantumAnsatz` to provide a kernel matrix."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.ansatz = QuantumAnsatz(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

# --------------------------------------------------------------------------- #
# 3.  Hybrid kernel
# --------------------------------------------------------------------------- #
class QuantumKernelMethod(nn.Module):
    """Weighted product of a quantum kernel and an RBF kernel."""
    def __init__(
        self,
        gamma: float = 1.0,
        q_weight: float = 0.5,
        n_wires: int = 4,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma=gamma, normalize=normalize)
        self.qk = QuantumKernel(n_wires=n_wires)
        self.q_weight = torch.tensor(q_weight, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf = self.rbf(x, y)
        qk = self.qk(x, y)
        return (self.q_weight * qk) + ((1.0 - self.q_weight) * rbf)

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        X = torch.stack(X)
        Y = torch.stack(Y)
        return self.forward(X, Y).detach().cpu().numpy()

    @staticmethod
    def tune(
        X: Sequence[torch.Tensor],
        Y: Sequence[torch.Tensor],
        gamma_grid: Tuple[float,...] = (0.1, 1.0, 10.0),
        q_weight_grid: Tuple[float,...] = (0.0, 0.5, 1.0),
    ) -> Tuple[float, float]:
        best_score = -np.inf
        best_params = (gamma_grid[0], q_weight_grid[0])
        for gamma in gamma_grid:
            for q_weight in q_weight_grid:
                model = QuantumKernelMethod(gamma=gamma, q_weight=q_weight)
                K = model.kernel_matrix(X, Y)
                score = np.trace(K)
                if score > best_score:
                    best_score = score
                    best_params = (gamma, q_weight)
        return best_params

__all__ = ["QuantumKernelMethod", "RBFKernel", "QuantumKernel"]
