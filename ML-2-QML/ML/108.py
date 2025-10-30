import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerQNNGen(nn.Module):
    """
    A richer classical sampler network.

    The architecture consists of:
    - 3 hidden layers with ReLU activations
    - BatchNorm and Dropout for regularisation
    - Output layer producing a probability vector via softmax

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector (default 2)
    hidden_dims : Tuple[int,...]
        Sizes of the hidden layers (default (8, 8, 8))
    dropout : float
        Dropout probability applied after each hidden layer (default 0.1)
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (8, 8, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the output distribution."""
        return 2

__all__ = ["SamplerQNNGen"]
