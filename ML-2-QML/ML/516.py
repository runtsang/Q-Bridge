import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

__all__ = ["FCL"]

class FCL(nn.Module):
    """
    Multi‑layer fully‑connected block that accepts a flat list of
    parameters.  Each layer uses a weight matrix of shape
    (n_features, 1) and applies a tanh activation.  Optional
    dropout and batch‑norm are supported to improve generalisation.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_layers: int = 1,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm_layer = nn.BatchNorm1d(n_features) if batchnorm else None

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network with a flat list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Length must be `n_features * n_layers`.  Each slice of
            length `n_features` is reshaped into a column vector
            that acts as the weight matrix for a single linear
            transformation.

        Returns
        -------
        np.ndarray
            The mean activation of the final layer as a 1‑D array.
        """
        thetas = torch.as_tensor(list(thetas), dtype=torch.float32)
        # reshape into (n_layers, n_features, 1)
        weights = thetas.view(self.n_layers, self.n_features, 1)

        # initial input: vector of ones
        x = torch.ones(self.n_features, dtype=torch.float32)

        for w in weights:
            x = torch.tanh(torch.matmul(x, w))
            if self.batchnorm_layer:
                x = self.batchnorm_layer(x)
            if self.dropout_layer:
                x = self.dropout_layer(x)

        expectation = x.mean()
        return expectation.detach().numpy()
