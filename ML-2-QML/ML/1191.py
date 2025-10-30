import torch
from torch import nn
import numpy as np
from typing import Iterable, Sequence

class HybridFCL(nn.Module):
    """
    Multi‑layer fully‑connected network that mimics the behaviour of a quantum
    fully‑connected layer but with classical weights.
    The network supports configurable hidden sizes and dropout, enabling
    richer feature extraction while keeping the API identical to the
    original `FCL` seed.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts a sequence of parameters (interpreted as input features),
        runs the network and returns a NumPy array of the mean activation.
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(x).mean(dim=0)
        return out.detach().cpu().numpy()

def FCL() -> HybridFCL:
    """Compatibility wrapper returning a default instance."""
    return HybridFCL()

__all__ = ["HybridFCL", "FCL"]
