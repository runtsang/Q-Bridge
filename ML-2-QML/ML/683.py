"""Hybrid kernel utilities with classical RBF and quantum augmentation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector


class RBFKernel(nn.Module):
    """Classical RBF kernel with optional input normalization."""
    def __init__(self, gamma: float = 1.0, normalize: bool = True):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel matrix between two batches."""
        if self.normalize:
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)
            y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-8)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [n_x, n_y, d]
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))


class QuantumKernelSampler:
    """Stochastic quantum kernel estimator using Qiskit Aer statevector simulator."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("statevector_simulator")

    def _encode(self, circuit: QuantumCircuit, data: np.ndarray):
        """Encode a classical data vector into a quantum circuit with RY rotations."""
        for i, val in enumerate(data):
            circuit.ry(val, i)

    def _kernel_value(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the overlap between two encoded states."""
        qc_x = QuantumCircuit(self.n_qubits)
        self._encode(qc_x, x)
        sv_x = execute(qc_x, self.backend, shots=self.shots).result().get_statevector()

        qc_y = QuantumCircuit(self.n_qubits)
        self._encode(qc_y, y)
        sv_y = execute(qc_y, self.backend, shots=self.shots).result().get_statevector()

        return float(np.abs(np.vdot(sv_x, sv_y)) ** 2)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Batch compute quantum kernel values."""
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        batch = []
        for xi in x_np:
            row = [self._kernel_value(xi, yi) for yi in y_np]
            batch.append(row)
        return torch.tensor(batch, dtype=torch.float32)


class HybridKernel(nn.Module):
    """Hybrid kernel combining a classical RBF kernel and a quantum kernel."""
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5, n_qubits: int = 4, shots: int = 1024):
        super().__init__()
        self.rbf = RBFKernel(gamma=gamma)
        self.qsampler = QuantumKernelSampler(n_qubits=n_qubits, shots=shots)
        self.alpha = alpha  # weight for quantum contribution

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a weighted sum of classical and quantum kernel values."""
        classical = self.rbf(x, y)
        quantum = self.qsampler(x, y)
        return (1 - self.alpha) * classical + self.alpha * quantum


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0, alpha: float = 0.5) -> np.ndarray:
    """Compute the Gram matrix between datasets `a` and `b` using HybridKernel."""
    kernel = HybridKernel(gamma=gamma, alpha=alpha)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["RBFKernel", "QuantumKernelSampler", "HybridKernel", "kernel_matrix"]
