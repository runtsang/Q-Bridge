"""Hybrid kernel implementation combining classical RBF and quantum overlap kernels.

This module extends the original `QuantumKernelMethod.py` by fusing
the classical radial‑basis‑function (RBF) kernel with a parametric
quantum kernel realized with TorchQuantum.  The two kernels are linearly
mixed in `HybridKernel` so that the user can tune the relative
contribution of classical versus quantum features via the `alpha`
parameter.  The `kernel_matrix` helper returns a NumPy array that can be
used directly in scikit‑learn estimators such as ``SVR(kernel='precomputed')``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence
import torchquantum as tq


class RBFKernel(nn.Module):
    """Standard RBF kernel implemented in PyTorch."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel based on a random layer followed by trainable
    rotations. The kernel value is the absolute overlap of the two
    encoded states.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"]
        )
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value k(x, y) = |⟨x | y⟩|."""
        # Encode first vector
        self.encoder(self.q_device, x)
        self.random_layer(self.q_device)
        for w in range(self.n_wires):
            self.rx(self.q_device, wires=w)
            self.ry(self.q_device, wires=w)
        # Encode second vector with opposite parameters
        self.encoder(self.q_device, -y)
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridKernel(nn.Module):
    """Linear combination of a classical RBF kernel and a quantum kernel.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel bandwidth.
    n_wires : int, optional
        Number of qubits used in the quantum circuit.
    alpha : float, optional
        Weight for the classical RBF component (0 ≤ alpha ≤ 1).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        n_wires: int = 4,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.qk = QuantumKernel(n_wires)
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.rbf(x, y) + (1.0 - self.alpha) * self.qk(x, y)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    n_wires: int = 4,
    alpha: float = 0.5,
) -> np.ndarray:
    """Return the Gram matrix using the hybrid kernel."""
    kernel = HybridKernel(gamma=gamma, n_wires=n_wires, alpha=alpha)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
#  Dataset utilities (borrowed from QuantumRegression.py)
# --------------------------------------------------------------------------- #

def generate_superposition_data(
    num_features: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data from a superposition‑like function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Simple feed‑forward network used as a stand‑in for a quantum head."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = [
    "RBFKernel",
    "QuantumKernel",
    "HybridKernel",
    "kernel_matrix",
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
]
