"""Enhanced fully connected layer with support for multiple layers, dropout, and batch normalization.

This class extends the original seed by allowing arbitrary depth, dropout, and batch normalization.
It exposes a ``run`` method that accepts a list or NumPy array of parameters and returns the
network output as a NumPy array, mirroring the API of the original ``FCL`` function.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Tuple, Union

class FullyConnectedLayer(nn.Module):
    """
    A flexible fully connected neural network that can be used as a classical surrogate
    for quantum layers. Supports arbitrary depth, dropout, and batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int,...], None] = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : list[int] | tuple[int,...] | None
            Sizes of hidden layers. If None, a single linear layer is used.
        output_dim : int
            Size of the output layer.
        dropout : float
            Dropout probability applied after each hidden layer (if > 0).
        use_batchnorm : bool
            Whether to apply batch normalization after linear layers.
        activation : nn.Module
            Activation function applied after each hidden layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: List[nn.Module] = []

        if hidden_dims is None:
            hidden_dims = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience method that accepts an iterable of parameters and returns the
        network's output as a NumPy array. This mirrors the API of the original seed.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameters with shape (batch_size, input_dim).

        Returns
        -------
        np.ndarray
            Output tensor as a NumPy array of shape (batch_size, output_dim).
        """
        theta_arr = np.array(list(thetas), dtype=np.float32)
        if theta_arr.ndim == 1:
            theta_arr = theta_arr.reshape(1, -1)
        with torch.no_grad():
            input_tensor = torch.from_numpy(theta_arr).float()
            output = self.forward(input_tensor)
            return output.numpy()
