import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridKernelSampler(nn.Module):
    """
    Classical hybrid kernel + sampler network.
    Combines a learnable radial basis function kernel with an optional
    pre‑computed quantum kernel matrix and a lightweight neural sampler.
    """
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5):
        """
        Parameters
        ----------
        gamma : float
            Width parameter of the RBF kernel.
        alpha : float
            Mixing coefficient between classical (alpha) and quantum (1-alpha) kernels.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sampler = self._build_sampler()

    def _build_sampler(self) -> nn.Sequential:
        """Small two‑layer MLP used as a classical sampler."""
        return nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel value between two samples."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor,
                      quantum_matrix: np.ndarray | None = None) -> np.ndarray:
        """
        Compute a hybrid kernel matrix.
        If `quantum_matrix` is supplied, it is combined with the classical
        RBF kernel using the weighting coefficient `alpha`.
        """
        a = a.view(-1, a.shape[-1])
        b = b.view(-1, b.shape[-1])
        # Classical RBF kernel
        classical = torch.exp(-self.gamma * torch.sum((a[:, None, :] - b[None, :, :]) ** 2, dim=-1))
        if quantum_matrix is None:
            return classical.numpy()
        quantum = torch.tensor(quantum_matrix, dtype=torch.float32)
        hybrid = self.alpha * classical + (1 - self.alpha) * quantum
        return hybrid.numpy()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sampler network.
        Returns a probability distribution over the two output classes.
        """
        logits = self.sampler(inputs)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridKernelSampler"]
