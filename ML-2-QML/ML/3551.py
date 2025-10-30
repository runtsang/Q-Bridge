import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class SamplerQNNGen177(nn.Module):
    """
    Classical hybrid sampler.
    Encodes input data into quantum circuit parameters and provides
    a classical RBF kernel for similarity evaluation.
    """
    def __init__(self, hidden_dim: int = 8, output_dim: int = 2, gamma: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim * 2)  # split into input & weight params
        )
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Produce a dictionary with keys 'input_params' and 'weight_params'.
        """
        out = self.net(inputs)
        input_params = torch.tanh(out[:, :2])   # 2 input params
        weight_params = torch.tanh(out[:, 2:])  # 4 weight params
        return {"input_params": input_params, "weight_params": weight_params}

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of vectors.
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (n, m, d)
        sq_norm = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Build Gram matrix using the RBF kernel.
        """
        x = torch.stack(a)
        y = torch.stack(b)
        return self.rbf_kernel(x, y).detach().cpu().numpy()

__all__ = ["SamplerQNNGen177"]
