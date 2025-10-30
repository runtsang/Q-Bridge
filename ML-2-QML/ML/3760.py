"""Hybrid kernel and classical estimator utilities."""
import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Optional

class ClassicalRBFKernel(nn.Module):
    """Gaussian radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridKernel(nn.Module):
    """
    Combines a classical RBF kernel with a quantum kernel.
    Parameters
    ----------
    gamma : float
        Width of the RBF kernel.
    quantum_weight : float, optional
        Weight given to the quantum kernel in the combined similarity.
        If None, only the classical kernel is used.
    qml_module : str, optional
        Python module containing a function ``quantum_kernel``.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 quantum_weight: Optional[float] = None,
                 qml_module: str = "qml_quantum_kernel") -> None:
        super().__init__()
        self.rbf = ClassicalRBFKernel(gamma)
        self.quantum_weight = quantum_weight
        if quantum_weight is not None:
            mod = __import__(qml_module, fromlist=["quantum_kernel"])
            self.quantum_kernel = mod.quantum_kernel
        else:
            self.quantum_kernel = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self.rbf(x, y)
        if self.quantum_kernel is None:
            return rbf_val.squeeze()
        q_val = torch.tensor(self.quantum_kernel(x.numpy(), y.numpy()),
                             dtype=rbf_val.dtype,
                             device=rbf_val.device)
        combined = (1.0 - self.quantum_weight) * rbf_val + self.quantum_weight * q_val
        return combined.squeeze()

def kernel_matrix(a: Sequence[torch.Tensor],
                 b: Sequence[torch.Tensor],
                 gamma: float = 1.0,
                 quantum_weight: Optional[float] = None,
                 qml_module: str = "qml_quantum_kernel") -> np.ndarray:
    """
    Compute Gram matrix between two sets of feature vectors using the hybrid kernel.
    """
    kernel = HybridKernel(gamma, quantum_weight, qml_module)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class EstimatorQNN(nn.Module):
    """Simple fully‑connected regressor for post‑processing kernel outputs."""
    def __init__(self, input_dim: int = 1, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

__all__ = ["ClassicalRBFKernel", "HybridKernel", "kernel_matrix", "EstimatorQNN"]
