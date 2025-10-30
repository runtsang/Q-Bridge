"""Hybrid classical model integrating RBF kernel, simulated quantum kernel, QCNN and SamplerQNN."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ClassicalRBFKernel(nn.Module):
    """Purely classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumKernelSim(nn.Module):
    """A lightweight, classical proxy for a quantum kernel using a parameterised linear map."""
    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.map = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inner‑product kernel on the mapped features."""
        fx, fy = self.forward(x), self.forward(y)
        return torch.sum(fx * fy, dim=-1, keepdim=True)


class QCNNModel(nn.Module):
    """Classical approximation of a QCNN that mirrors the quantum architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class SamplerQNN(nn.Module):
    """Simple neural sampler mirroring the quantum SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class HybridKernelQCNNQML(nn.Module):
    """Hybrid model that combines classical RBF, simulated quantum kernel, QCNN and SamplerQNN."""
    def __init__(self, dim: int, rbf_gamma: float = 1.0, quantum_weight: float = 0.5) -> None:
        super().__init__()
        self.rbf = ClassicalRBFKernel(rbf_gamma)
        self.quantum = QuantumKernelSim(dim)
        self.qcnn = QCNNModel()
        self.sampler = SamplerQNN()
        self.quantum_weight = quantum_weight

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute weighted sum of classical RBF and simulated quantum kernels."""
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        # Classical RBF
        rbf_mat = np.array([[self.rbf(x, y).item() for y in b_t] for x in a_t])
        # Quantum kernel
        q_mat = np.array([[self.quantum.kernel(x, y).item() for y in b_t] for x in a_t])
        return self.quantum_weight * q_mat + (1 - self.quantum_weight) * rbf_mat

    def qcnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the QCNN approximation."""
        return self.qcnn(x)

    def sampler_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sampler network."""
        return self.sampler(x)

__all__ = ["HybridKernelQCNNQML", "ClassicalRBFKernel", "QuantumKernelSim", "QCNNModel", "SamplerQNN"]
