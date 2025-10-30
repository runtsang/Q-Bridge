import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Sequence

class FullyConnectedLayer(nn.Module):
    """
    Multi‑layer fully connected network with optional batch‑norm and dropout.
    Designed to mirror the interface of the original FCL while providing richer
    expressiveness for downstream experiments.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Sequence[int] = (32, 16),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Replace all learnable parameters with the flat vector `thetas` and
        perform a forward pass on a unit‑input batch.  The method returns
        a NumPy array to match the original seed's API.
        """
        # Flatten current parameters
        params = torch.cat([p.view(-1) for p in self.parameters()]).detach()
        thetas = torch.as_tensor(thetas, dtype=params.dtype)

        # Sanity check
        if thetas.numel()!= params.numel():
            raise ValueError(f"Expected {params.numel()} parameters, got {thetas.numel()}")

        # Load new parameters
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(thetas[offset:offset + numel].view_as(p))
            offset += numel

        # Forward pass on a dummy input
        with torch.no_grad():
            dummy = torch.ones(1, self.net[0].in_features)
            out = self.forward(dummy)
        return out.detach().cpu().numpy()

def FCL() -> FullyConnectedLayer:
    """
    Public factory returning the fully‑connected layer instance.
    """
    return FullyConnectedLayer()

__all__ = ["FCL"]
