import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu, dropout
from typing import Iterable

class FCL(nn.Module):
    """
    Extended fully connected layer with residual connections and dropout.
    Accepts a list of input values via run(thetas) and returns the
    mean activation as a NumPy array, mirroring the original seed.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 32,
                 n_layers: int = 3, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.linear_layers = nn.ModuleList()
        # First linear layer
        self.linear_layers.append(nn.Linear(n_features, hidden_dim))
        # Hidden layers
        for _ in range(n_layers - 2):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        self.linear_layers.append(nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, linear in enumerate(self.linear_layers):
            if i < self.n_layers - 1:
                residual = out
                out = linear(relu(out))
                out = out + residual  # skip connection
                out = self.dropout(out)
            else:
                out = linear(out)
        return out

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimics the original seed's run signature: accepts an iterable
        of input values, converts them to a tensor, feeds them through
        the network, and returns the mean activation as a NumPy array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            output = self.forward(values)
            expectation = torch.mean(output, dim=0)
        return expectation.detach().cpu().numpy()
