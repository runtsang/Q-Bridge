import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Tuple

class RBFKernelAnsatz(nn.Module):
    """Efficient RBF kernel for batched tensors."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Pairwise squared Euclidean distance
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

class ClassicalKernel(nn.Module):
    """Wrapper around :class:`RBFKernelAnsatz` producing a Gram matrix."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = RBFKernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

class ClassicalAttention(nn.Module):
    """Self‑attention block operating on classical feature vectors."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # Linear projections
        q = self.w_q(inputs)
        k = self.w_k(inputs)
        v = self.w_v(inputs)
        # Scale query
        q = q / np.sqrt(self.embed_dim)
        # Attention scores
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        # Weighted sum
        return torch.matmul(scores, v)

class HybridKernelAttention(nn.Module):
    """Combines a kernel matrix with self‑attention for downstream tasks."""
    def __init__(self,
                 kernel_gamma: float = 1.0,
                 embed_dim: int = 4):
        super().__init__()
        self.kernel = ClassicalKernel(kernel_gamma)
        self.attention = ClassicalAttention(embed_dim)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # Compute kernel Gram matrix
        K = self.kernel(x, y)
        # Compute attention output
        A = self.attention(x, rotation_params, entangle_params)
        # Simple fusion: element‑wise product
        return K * A

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Return Gram matrix between two collections of tensors."""
    x = torch.stack(a)
    y = torch.stack(b)
    kernel = ClassicalKernel(gamma)
    return kernel(x, y).detach().cpu().numpy()

def attention_matrix(inputs: torch.Tensor,
                     rotation_params: torch.Tensor,
                     entangle_params: torch.Tensor) -> np.ndarray:
    """Compute self‑attention scores for a batch of inputs."""
    attn = ClassicalAttention(inputs.shape[-1])
    return attn(inputs, rotation_params, entangle_params).detach().cpu().numpy()

def hybrid_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  rotation_params: torch.Tensor,
                  entangle_params: torch.Tensor,
                  gamma: float = 1.0) -> np.ndarray:
    """Hybrid kernel‑attention matrix."""
    x = torch.stack(a)
    y = torch.stack(b)
    hybrid = HybridKernelAttention(gamma, inputs.shape[-1])
    return hybrid(x, y, rotation_params, entangle_params).detach().cpu().numpy()

# Back‑compatibility aliases
KernalAnsatz = RBFKernelAnsatz
Kernel = ClassicalKernel
__all__ = [
    "RBFKernelAnsatz",
    "ClassicalKernel",
    "ClassicalAttention",
    "HybridKernelAttention",
    "kernel_matrix",
    "attention_matrix",
    "hybrid_matrix",
    "KernalAnsatz",
    "Kernel",
]
