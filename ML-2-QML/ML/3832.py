"""Hybrid kernel ridge regression with classical RBF and TorchQuantum kernel.

The class :class:`HybridQuantumKernelRegressor` implements a kernel ridge
regressor that uses a **hybrid kernel**: the element‑wise product of
a classical RBF kernel and a lightweight quantum kernel built with
TorchQuantum.  The implementation is fully tensorized so that it can
run on GPU via PyTorch, while the quantum kernel is evaluated on
a CPU simulator for simplicity.  The class also exposes a small
feed‑forward network (:class:`EstimatorNN`) that can be swapped in
for experiments that require a trainable mapping.
"""

import torch
import numpy as np
import torchquantum as tq
from torch import nn
from torchquantum.functional import func_name_dict
from typing import Sequence

# --------------------------------------------------------------------------- #
# Classical RBF kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """TorchTensor implementation of a Gaussian RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Width parameter of the Gaussian.  Default is ``1.0``.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        # Sum over feature dimension, keep batch dimension
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Quantum kernel based on a programmable list of gates
# --------------------------------------------------------------------------- #
class QuantumAnsatz(tq.QuantumModule):
    """Encodes data through a list of parameterised Ry gates."""
    def __init__(self, gate_list: Sequence[dict]) -> None:
        super().__init__()
        self.gate_list = gate_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode ``x`` and ``y`` into the same device and apply the inverse."""
        q_device.reset_states(x.shape[0])
        for info in self.gate_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.gate_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Wrapper that evaluates the quantum kernel."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the real part of the amplitude of |0…0>
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# Hybrid kernel ridge regressor
# --------------------------------------------------------------------------- #
class HybridQuantumKernelRegressor(nn.Module):
    """Kernel ridge regressor using a hybrid classical‑quantum kernel.

    The kernel is ``K(x, y) = RBF(x, y) * QuantumKernel(x, y)``.
    The regression coefficients are computed in closed form:
    α = (K + λI)⁻¹ y.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        n_wires: int = 4,
        reg: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reg = reg
        self.device = device

        self.rbf = KernalAnsatz(gamma).to(device)
        self.quantum = QuantumKernel(n_wires).to(device)

        # Placeholders for training data and coefficients
        self.X_train: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    # Helper: compute hybrid kernel matrix in a vectorised manner
    # --------------------------------------------------------------------- #
    def _hybrid_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return Gram matrix K_{ij} = RBF(X[i], Y[j]) * QuantumKernel(X[i], Y[j])."""
        # Classical RBF
        K_rbf = self.rbf(X, Y).squeeze(-1)  # shape (len(X), len(Y))

        # Quantum kernel; TorchQuantum expects batches
        K_q = torch.zeros_like(K_rbf)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K_q[i, j] = self.quantum(x.unsqueeze(0), y.unsqueeze(0))
        return K_rbf * K_q

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def fit(self, X: Sequence[torch.Tensor], y: Sequence[float]) -> None:
        """
        Fit the model by computing the hybrid kernel matrix and solving
        the regularised linear system for the coefficients.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
            Training data.
        y : array‑like, shape (n_samples,)
            Target values.
        """
        self.X_train = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(-1)

        K = self._hybrid_kernel_matrix(self.X_train, self.X_train)
        I = self.reg * torch.eye(K.shape[0], device=self.device)
        self.alpha = torch.linalg.solve(K + I, y)

    def predict(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Predict on new data.

        Parameters
        ----------
        X : array‑like, shape (n_test, n_features)

        Returns
        -------
        predictions : ndarray, shape (n_test,)
        """
        if self.X_train is None or self.alpha is None:
            raise RuntimeError("Model has not been fitted.")
        X_test = torch.tensor(X, dtype=torch.float32, device=self.device)
        K_test = self._hybrid_kernel_matrix(X_test, self.X_train)
        y_pred = K_test @ self.alpha
        return y_pred.squeeze().cpu().numpy()

# --------------------------------------------------------------------------- #
# Estimator network that can be used as a post‑processing layer
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor that can be attached to the hybrid
    kernel regression for experiments that require a trainable mapping.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# Convenience helper to compute the kernel matrix as a NumPy array
# --------------------------------------------------------------------------- #
def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    n_wires: int = 4,
) -> np.ndarray:
    """Return the hybrid kernel matrix between ``a`` and ``b``."""
    regressor = HybridQuantumKernelRegressor(gamma, n_wires)
    return regressor._hybrid_kernel_matrix(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32)
    ).cpu().numpy()

__all__ = [
    "KernalAnsatz",
    "QuantumKernel",
    "HybridQuantumKernelRegressor",
    "EstimatorNN",
    "kernel_matrix",
]
