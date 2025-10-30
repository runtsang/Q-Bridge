import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz compatible with quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` to provide a 2‑D kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridSamplerKernel(nn.Module):
    """
    Classical sampler that uses an RBF kernel to embed inputs into a feature space
    and a lightweight feed‑forward network to produce a two‑class probability
    distribution.  The architecture is deliberately simple to keep training
    fast while still demonstrating the benefit of kernel‑based feature
    engineering.
    """
    def __init__(self, gamma: float = 1.0, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.kernel = Kernel(gamma)
        self.hidden_dim = hidden_dim
        self.sampler = None

    def _build_sampler(self, num_refs: int) -> None:
        hidden = self.hidden_dim or num_refs
        self.sampler = nn.Sequential(
            nn.Linear(num_refs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of input vectors of shape (B, D).
        references : torch.Tensor
            Reference vectors of shape (R, D) used to compute kernel similarities.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (B, 2).
        """
        # Compute kernel similarities, shape (B, R)
        k_mat = torch.tensor(kernel_matrix(x, references, self.kernel.gamma))
        if self.sampler is None:
            self._build_sampler(k_mat.shape[1])
        probs = F.softmax(self.sampler(k_mat), dim=-1)
        return probs

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "HybridSamplerKernel"]
