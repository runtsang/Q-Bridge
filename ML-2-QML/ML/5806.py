"""
HybridFCL: a classical fully connected layer with optional depth, mirroring quantum structure.
"""

import torch
from torch import nn
from typing import Iterable, List

class HybridFCL(nn.Module):
    """Classic feed‑forward network that emulates a quantum‑style fully‑connected layer.

    Parameters
    ----------
    n_features : int
        Number of input features / qubits.
    depth : int, default 1
        How many hidden layers to stack.
    """
    def __init__(self, n_features: int, depth: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, n_features))
            layers.append(nn.ReLU())
            in_dim = n_features
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the mean of the tanh activation."""
        out = torch.tanh(self.network(x))
        return out.mean(dim=1, keepdim=True)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Run with a flat list of parameters, reshaping them to match the network."""
        params = torch.tensor(list(thetas), dtype=torch.float32)
        expected = sum(p.numel() for p in self.parameters())
        if params.numel()!= expected:
            raise ValueError(f"Expected {expected} parameters, got {params.numel()}")
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(params[idx:idx + size].view_as(p))
            idx += size
        x = params.view(-1, 1)
        return self.forward(x).detach().numpy()

__all__ = ["HybridFCL"]
