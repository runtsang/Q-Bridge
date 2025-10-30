import math
import numpy as np
import torch
from torch import nn

class QuantumFullyConnectedKernel(nn.Module):
    """
    Hybrid classical module that merges a fully‑connected layer and a trainable
    RBF kernel.  The fully‑connected part is a linear layer followed by a tanh
    non‑linearity, mirroring the behaviour of the original quantum FCL.  The
    RBF kernel is implemented in PyTorch and accepts a trainable gamma
    parameter.  The class exposes a unified `forward` that returns both
    components and a `kernel_matrix` helper for Gram‑matrix computation.
    """
    def __init__(self,
                 n_features: int = 1,
                 gamma: float = 1.0,
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.device = device or torch.device('cpu')
        self.linear.to(self.device)
        self.gamma.to(self.device)

    def _fc_expectation(self, thetas: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the expectation of the linear layer."""
        if thetas is None:
            raise ValueError("`thetas` must be provided for the FC part.")
        x = thetas.view(-1, 1).float().to(self.device)
        out = torch.tanh(self.linear(x))
        return out.mean(dim=0)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single‑sample RBF kernel."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self,
                x: torch.Tensor,
                thetas: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, features).  Only the first feature
            is used for the kernel in this toy example.
        thetas : torch.Tensor, optional
            Parameters for the fully‑connected part.  When omitted the
            FC output is set to zero.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (fc_output, kernel_output)
        """
        fc_out = self._fc_expectation(thetas) if thetas is not None else torch.tensor(0.0, device=self.device)
        # Kernel between each sample and itself (diagonal of Gram matrix).
        k = self._rbf(x, x)
        return fc_out, k.squeeze(-1)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the Gram matrix between two sets of samples."""
        a = a.view(-1, a.shape[-1])
        b = b.view(-1, b.shape[-1])
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # (len(a), len(b), features)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

def FCL() -> QuantumFullyConnectedKernel:
    """Compatibility shim mirroring the original FCL interface."""
    return QuantumFullyConnectedKernel()

__all__ = ["QuantumFullyConnectedKernel", "FCL"]
