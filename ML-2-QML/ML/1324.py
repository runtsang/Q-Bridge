import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

class FCL(nn.Module):
    """A richer fully connected layer with optional batch norm and dropout.

    The layer can be used as a building block in hybrid models where the
    output is interpreted as a classical expectation value.
    """

    def __init__(self, n_features: int = 1, hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.tanh(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the layer on a sequence of theta values.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter values to feed into the layer.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the mean output over thetas.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.forward(values)
        mean = out.mean(dim=0)
        return mean.detach().cpu().numpy()
