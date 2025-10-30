import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    Classical fully connected layer with skip connections and batch normalization.
    The ``run`` method accepts a list of input features (thetas) and returns a single output.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Layers
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        x = x.float()
        out = F.relu(self.fc1(x))
        out = self.bn1(out)
        out = self.dropout_layer(out)

        # Skip connection from first hidden to second hidden
        out = out + F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout_layer(out)

        out = self.out(out)
        return out

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network on a list of input values.
        Parameters
        ----------
        thetas : Iterable[float]
            Input features for the network.
        Returns
        -------
        np.ndarray
            Output of the network as a 1D array.
        """
        with torch.no_grad():
            inputs = torch.tensor(thetas, dtype=torch.float32).unsqueeze(0)
            output = self.forward(inputs)
            return output.detach().cpu().numpy().flatten()

__all__ = ["FCL"]
