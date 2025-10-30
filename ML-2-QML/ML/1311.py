import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """Trainable RBF kernel for classical data."""
    def __init__(self, gamma: float = 1.0, trainable: bool = True):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma)) if trainable else torch.tensor(gamma)
        self.trainable = trainable

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure 2‑D tensors
        x = x if x.dim() == 2 else x.unsqueeze(0)
        y = y if y.dim() == 2 else y.unsqueeze(0)
        # Compute squared Euclidean distance
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (n, m, d)
        dist_sq = (diff ** 2).sum(-1)          # shape (n, m)
        return torch.exp(-self.gamma * dist_sq)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return Gram matrix between sets a and b."""
        return self.forward(a, b)
