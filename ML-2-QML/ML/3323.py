import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper for the RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute a Gram matrix using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class SelfAttentionHybrid:
    """Classical self‑attention that uses a quantum‑style kernel for weighting."""
    def __init__(self, embed_dim: int, gamma: float = 1.0) -> None:
        self.embed_dim = embed_dim
        self.kernel = Kernel(gamma)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        # Linear projections
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)

        # Quantum‑kernel style similarity between query and key
        kernel_mat = np.array([[self.kernel(q.unsqueeze(0), k.unsqueeze(0)).item() for k in key] for q in query])
        kernel_mat = kernel_mat / kernel_mat.sum(axis=1, keepdims=True)

        # Weighted sum of values
        output = kernel_mat @ value.numpy()
        return output

def SelfAttention() -> SelfAttentionHybrid:
    """Convenience factory matching the original API."""
    return SelfAttentionHybrid(embed_dim=4)

__all__ = ["SelfAttention", "SelfAttentionHybrid", "Kernel", "KernalAnsatz", "kernel_matrix"]
