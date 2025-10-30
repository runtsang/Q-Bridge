import numpy as np
import torch
from torch import nn
from typing import Iterable

class FCLExtended(nn.Module):
    """
    A configurable fully connected network that mimics the original FCL API.
    Parameters can be supplied as a flat iterable of floats, one for every
    weight and bias in the network. Supports an arbitrary number of hidden
    layers, user‑chosen activation (tanh or relu) and optional dropout.
    """
    def __init__(self,
                 n_features: int = 1,
                 hidden_sizes: Iterable[int] = (10,),
                 activation: str = "tanh",
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        layers = []
        prev_size = n_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))  # output layer
        self.layers = nn.ModuleList(layers)

    def _apply_params(self, thetas: Iterable[float]) -> None:
        """
        Load the flat list of parameters into the network weights and biases.
        """
        t_iter = iter(thetas)
        for layer in self.layers:
            weight_shape = layer.weight.shape
            bias_shape = layer.bias.shape
            weight = torch.tensor(
                [next(t_iter) for _ in range(weight_shape.numel())],
                dtype=torch.float32
            ).view(weight_shape)
            bias = torch.tensor(
                [next(t_iter) for _ in range(bias_shape.numel())],
                dtype=torch.float32
            ).view(bias_shape)
            layer.weight.data = weight
            layer.bias.data = bias

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the expectation value of the network output.
        """
        self._apply_params(thetas)
        x = torch.zeros((1, self.layers[0].in_features), dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
            if self.activation == "tanh":
                x = torch.tanh(x)
            elif self.activation == "relu":
                x = torch.relu(x)
            x = self.dropout(x)
        return x.mean().detach().numpy()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compatibility method mirroring the seed's API.
        """
        return self.forward(thetas)

__all__ = ["FCLExtended"]
