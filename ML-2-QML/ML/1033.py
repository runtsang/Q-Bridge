import torch
import torch.nn as nn
from typing import Sequence, Iterable

class FCL(nn.Module):
    """
    Classical fully connected layer with optional hidden layers and dropout.
    The network can be configured with arbitrary depth and width.
    """
    def __init__(self, input_dim: int = 1, hidden_dims: Sequence[int] = (), dropout: float = 0.0):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature vector.
        hidden_dims : Sequence[int]
            Sizes of hidden layers. An empty tuple yields a single linear layer.
        dropout : float
            Dropout probability applied after each hidden layer (if > 0).
        """
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Run the network with the supplied parameters as input features.
        The parameters are interpreted as a flat vector of input values.
        Returns the output as a 1‑D tensor.
        """
        input_tensor = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        return self.forward(input_tensor).squeeze(0)

    def parameters_flat(self) -> torch.Tensor:
        """Return all parameters as a flat 1‑D tensor."""
        return torch.cat([p.view(-1) for p in self.parameters()])

__all__ = ["FCL"]
