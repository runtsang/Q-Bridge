"""Enhanced fully connected neural network layer with multi‑layer support.

This module extends the original toy implementation by allowing:
* arbitrary depth (n_hidden layers)
* configurable activation functions
* dropout regularisation
* a convenient `run` interface that accepts a flat list of parameters
* a `forward` method suitable for autograd training

The class is intentionally lightweight so it can be dropped into a
scikit‑learn pipeline or a PyTorch training loop.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Callable, Optional

# Activation mapping to nn.Module instances
_ACTIVATIONS = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "none": nn.Identity(),
}

class FCL(nn.Module):
    """
    Fully‑connected neural network layer stack.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input.
    n_hidden : int, default 0
        Number of hidden layers. If 0, the network reduces to a single linear layer.
    hidden_dim : int, default 16
        Width of each hidden layer.
    activation : str, default 'tanh'
        Activation function applied after each linear transformation.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        n_features: int = 1,
        n_hidden: int = 0,
        hidden_dim: int = 16,
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers: List[nn.Module] = []
        act_fn: nn.Module = _ACTIVATIONS.get(activation, nn.Tanh())

        last_dim = n_features
        for _ in range(n_hidden):
            self.layers.append(nn.Linear(last_dim, hidden_dim))
            self.layers.append(act_fn)
            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        # Final output layer
        self.layers.append(nn.Linear(last_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x

    def _set_params_from_flat(self, flat_params: Iterable[float]) -> None:
        """Load parameters from a flat iterable into the module."""
        flat_params = np.asarray(list(flat_params), dtype=np.float32)
        idx = 0
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                weight_shape = layer.weight.shape
                bias_shape = layer.bias.shape
                weight_size = weight_shape[0] * weight_shape[1]
                bias_size = bias_shape[0]
                weight = flat_params[idx : idx + weight_size].reshape(weight_shape)
                idx += weight_size
                bias = flat_params[idx : idx + bias_size]
                idx += bias_size
                layer.weight.data = torch.from_numpy(weight)
                layer.bias.data = torch.from_numpy(bias)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the network on a single input vector defined by *thetas*.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of parameters that will be loaded into the network.

        Returns
        -------
        np.ndarray
            The mean of the network output across the batch.
        """
        self._set_params_from_flat(thetas)
        # Create a dummy input of shape (1, n_features)
        dummy = torch.ones((1, self.layers[0].in_features), dtype=torch.float32)
        out = self.forward(dummy)
        return out.mean().detach().numpy()

    def get_param_vector(self) -> np.ndarray:
        """Return a flattened copy of the network parameters."""
        params = []
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                params.append(layer.weight.detach().cpu().numpy().flatten())
                params.append(layer.bias.detach().cpu().numpy().flatten())
        return np.concatenate(params)

__all__ = ["FCL", "get_FCL"]

def get_FCL(
    n_features: int = 1,
    n_hidden: int = 0,
    hidden_dim: int = 16,
    activation: str = "tanh",
    dropout: float = 0.0,
) -> FCL:
    """Convenience constructor that mirrors the original API."""
    return FCL(
        n_features=n_features,
        n_hidden=n_hidden,
        hidden_dim=hidden_dim,
        activation=activation,
        dropout=dropout,
    )
