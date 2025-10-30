"""Hybrid classical‑quantum module combining a fully‑connected layer, a sampler network, and a quantum kernel."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """Small neural network that produces quantum variational parameters."""
    def __init__(self, in_features: int = 2, hidden: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output a flat vector of weights
        return self.net(x)


class QuantumKernelModule(nn.Module):
    """Placeholder for a quantum kernel; falls back to an RBF kernel for compatibility."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-0.5 * torch.sum(diff * diff, dim=-1, keepdim=True))


class FCLHybrid(nn.Module):
    """
    Hybrid fully‑connected layer that uses a classical sampler to generate
    parameters for a quantum circuit and evaluates a quantum kernel for
    similarity measurement.
    """
    def __init__(self, n_features: int = 1, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits

        # Classical sampler generating the variational parameters
        self.sampler = SamplerModule(in_features=2, hidden=4, out_features=2 * n_qubits)

        # Quantum kernel (placeholder; will be replaced by QML side)
        self.kernel = QuantumKernelModule(n_qubits)

        # Linear layer to map quantum expectation to output
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode input, generate parameters, simulate quantum circuit,
        and return a scalar output.
        """
        # generate quantum parameters
        theta = self.sampler(x)
        theta = theta.view(-1, self.n_qubits)

        # simulate quantum expectation via QML side (placeholder)
        expectation = torch.tanh(theta).mean(dim=1, keepdim=True)

        # linear mapping
        return self.linear(expectation)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Exposed API compatible with the original FCL example.
        ``thetas`` is expected to be a 1‑D array of length ``n_qubits``.
        """
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix between two datasets using the quantum kernel.
        """
        a_t = torch.from_numpy(a).float()
        b_t = torch.from_numpy(b).float()
        mat = torch.exp(-0.5 * torch.cdist(a_t, b_t, p=2) ** 2)
        return mat.numpy()


__all__ = ["FCLHybrid"]
