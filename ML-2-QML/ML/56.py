"""Enhanced fully connected layer with configurable depth and dropout."""
import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

class FCL(nn.Module):
    """Multi-layer fully connected neural network.

    Parameters
    ----------
    input_dim : int
        Size of input features.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    output_dim : int, optional
        Size of output layer, default 1.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Load parameters from ``thetas`` and compute the mean output
        for a fixed input of ones.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat sequence of all trainable parameters of the network.

        Returns
        -------
        np.ndarray
            Array containing the mean of the network output.
        """
        # Flatten current state dict keys to ensure order
        param_shapes = [p.shape for p in self.parameters()]
        param_sizes = [int(np.prod(s)) for s in param_shapes]
        if len(thetas)!= sum(param_sizes):
            raise ValueError(f"Expected {sum(param_sizes)} parameters, got {len(thetas)}")
        # Load parameters
        cursor = 0
        for p, size in zip(self.parameters(), param_sizes):
            flat = torch.tensor(list(thetas[cursor:cursor+size]), dtype=p.dtype, device=p.device)
            p.data = flat.view(p.shape)
            cursor += size
        # Fixed input: ones
        with torch.no_grad():
            dummy = torch.ones((1, self.net[0].in_features), dtype=torch.float32)
            out = self(dummy)
            mean_out = out.mean().item()
        return np.array([mean_out])

__all__ = ["FCL"]
