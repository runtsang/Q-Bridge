"""Enhanced fully connected layer with flexible architecture and dropout support."""

import numpy as np
import torch
from torch import nn
from typing import Iterable

class FullyConnectedLayerGen137(nn.Module):
    """A versatile fully connected block that supports multiple hidden layers,
    dropout, and custom activation functions. It can be used as a building
    block in larger neural networks or as a standalone module for quick
    experimentation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (64, 32),
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Accepts a flat list of parameters and returns the mean output."""
        with torch.no_grad():
            flat_params = torch.as_tensor(list(thetas), dtype=torch.float32)
            idx = 0
            for param in self.parameters():
                numel = param.numel()
                param.data.copy_(flat_params[idx : idx + numel].view(param.shape))
                idx += numel
            dummy = torch.randn(1, self.network[0].in_features)
            out = self.forward(dummy)
            return out.detach().cpu().numpy()

def FCL() -> FullyConnectedLayerGen137:
    """Factory that returns an instance of the extended fully connected layer."""
    return FullyConnectedLayerGen137(
        input_dim=1,
        hidden_dims=(128, 64),
        output_dim=1,
        dropout=0.1,
        activation="tanh",
    )

__all__ = ["FCL", "FullyConnectedLayerGen137"]
