"""Hybrid classical-quantum regressor combining a feed-forward network, a classical RBF kernel, and a quantum kernel.

The model extracts latent features via a small neural network, then evaluates both a classical
radial‑basis function kernel and a quantum kernel (implemented in :mod:`EstimatorQNN__gen209_qml`)
between the latent representation and a set of trainable support vectors.  The two kernels are
averaged and linearly combined to produce the final prediction.  This architecture allows the
classical network to learn useful feature embeddings while the quantum kernel injects
non‑linear quantum correlations that are otherwise difficult to capture classically.
"""

from __future__ import annotations

import torch
from torch import nn

# Import the quantum kernel implementation from the QML module
try:
    from EstimatorQNN__gen209_qml import EstimatorQNNHybrid as QuantumKernel
except Exception:
    # Fallback stub if the QML module is not available
    class QuantumKernel(nn.Module):
        def __init__(self, *_, **__):
            super().__init__()
        def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], y.shape[0], device=x.device)

class RBFKernel(nn.Module):
    """Simple radial‑basis function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim), y: (support, dim)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (batch, support, dim)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

class EstimatorQNNHybrid(nn.Module):
    """Hybrid regressor that combines classical and quantum kernels."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        latent_dim: int = 4,
        n_support: int = 10,
        gamma: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(device)

        # Trainable support vectors in latent space
        self.support_vectors = nn.Parameter(torch.randn(n_support, latent_dim, device=device))

        # Kernel modules
        self.rbf_kernel = RBFKernel(gamma).to(device)
        self.quantum_kernel = QuantumKernel().to(device)

        # Linear combination weights
        self.weights = nn.Parameter(torch.randn(n_support, 1, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch,).
        """
        latent = self.feature_extractor(x.to(self.device))
        # Classical RBF kernel
        k_classical = self.rbf_kernel(latent, self.support_vectors)  # (batch, n_support)

        # Quantum kernel
        # Compute kernel matrix by evaluating the quantum kernel for each pair
        k_quantum = torch.stack([
            torch.stack([self.quantum_kernel.kernel(l, sv) for sv in self.support_vectors])
            for l in latent
        ], dim=0)  # (batch, n_support)

        # Simple average of kernels
        k_combined = 0.5 * k_classical + 0.5 * k_quantum
        # Linear readout
        out = torch.matmul(k_combined, self.weights).squeeze(-1)
        return out

def EstimatorQNN() -> EstimatorQNNHybrid:
    """Convenience factory matching the original API."""
    return EstimatorQNNHybrid()
