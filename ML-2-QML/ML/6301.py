import torch
import torch.nn as nn
import numpy as np

class ClassicalRBF(nn.Module):
    """Pure classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridKernel(nn.Module):
    """
    Hybrid kernel that can operate in classical RBF mode or a classical approximation
    of a quantum kernel. The `mode` argument selects the behaviour.
    """
    def __init__(self, mode: str = "classical", gamma: float = 1.0, n_wires: int = 4):
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        if mode == "classical":
            self.kernel = ClassicalRBF(gamma)
        else:
            # classical surrogate for a quantum RY‑encoding kernel
            self.n_wires = n_wires
            self.width = gamma / (n_wires + 1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "classical":
            return self.kernel(x, y)
        diff = x - y
        return torch.exp(-self.width * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Return the Gram matrix between two sets of samples."""
        a = a.reshape(-1, a.shape[-1]) if a.ndim == 2 else a
        b = b.reshape(-1, b.shape[-1]) if b.ndim == 2 else b
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

class EstimatorQNN(nn.Module):
    """Tiny feed‑forward regressor that mirrors the EstimatorQNN example."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

__all__ = ["HybridKernel", "EstimatorQNN"]
