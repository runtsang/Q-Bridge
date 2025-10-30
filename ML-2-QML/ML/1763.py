import torch
import torch.nn as nn
import numpy as np
from typing import Sequence


class QuantumKernel(nn.Module):
    """
    Classical RBF kernel with learnable, data‑dependent scaling.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    hidden_dim : int, default=32
        Width of the hidden layer in the gamma‑predicting MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # guarantees positivity
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between two tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (input_dim,).  The method also accepts
            batched inputs of shape (batch, input_dim) and performs
            broadcasting appropriately.

        Returns
        -------
        torch.Tensor
            Kernel value(s) of shape (batch_x, batch_y).
        """
        # Ensure inputs are 2‑D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # Adaptive gamma per sample
        gamma_x = self.gamma_net(x).unsqueeze(-1)  # (batch_x, 1)
        gamma_y = self.gamma_net(y).unsqueeze(-1)  # (batch_y, 1)
        gamma = (gamma_x + gamma_y.T) / 2  # broadcast to (batch_x, batch_y)

        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (batch_x, batch_y, dim)
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)  # (batch_x, batch_y, 1)
        return torch.exp(-gamma * sq_norm).squeeze(-1)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  kernel: QuantumKernel) -> np.ndarray:
    """
    Compute the Gram matrix between two lists of tensors using the given kernel.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        First list of input vectors.
    b : Sequence[torch.Tensor]
        Second list of input vectors.
    kernel : QuantumKernel
        Instance of the adaptive RBF kernel.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    mat = torch.stack([kernel(x, y) for x in a for y in b])
    mat = mat.view(len(a), len(b))
    return mat.detach().cpu().numpy()


__all__ = ["QuantumKernel", "kernel_matrix"]
