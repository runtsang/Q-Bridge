import torch
import numpy as np
from torch import nn
from typing import Iterable, Sequence

class FCL(nn.Module):
    """
    Classical fully connected layer with optional hidden layers, dropout and
    batch‑normalisation.  The ``run`` method keeps the original API: it accepts
    an iterable of floats (the “theta” values) and returns the mean
    expectation value of the network output for those inputs.
    """
    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(n_features, hidden_sizes[0]), activation()]
        for i in range(len(hidden_sizes) - 1):
            layers += [
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                activation(),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.Dropout(dropout),
            ]
        layers += [nn.Linear(hidden_sizes[-1], 1), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the network on a batch of scalar inputs.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of scalar input values.

        Returns
        -------
        np.ndarray
            1‑D array containing the mean output of the network over the
            supplied inputs.
        """
        # Convert to a column tensor
        values = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            outputs = self.forward(values)
        expectation = outputs.mean(dim=0).item()
        return np.array([expectation])

__all__ = ["FCL"]
