"""FullyConnectedLayer: Classical multi‑layer perceptron with dropout and batch norm.

This class is a drop‑in replacement for the original FCL, but supports multiple hidden layers,
dropout, and batch normalization. The ``run`` method accepts a flat list of parameters
corresponding to all linear layers and returns the mean activation after the final layer.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """
    Multi‑layer perceptron with optional dropout and batch normalization.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_layers : List[int], optional
        Sizes of hidden layers. Default is one hidden layer of size 32.
    dropout_rate : float, optional
        Dropout probability applied after each hidden layer. 0 disables dropout.
    """

    def __init__(
        self,
        n_features: int,
        hidden_layers: List[int] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [32]
        layers: List[nn.Module] = []

        in_features = n_features
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass using a flat list of parameters.

        The parameters are expected in the order of all linear layers in the network.
        The list is reshaped to match each layer's weight and bias tensors.
        """
        thetas = torch.tensor(list(thetas), dtype=torch.float32)
        idx = 0
        for module in self.model:
            if isinstance(module, nn.Linear):
                # Flatten weight and bias
                weight_shape = module.weight.shape
                bias_shape = module.bias.shape
                weight_size = weight_shape.numel()
                bias_size = bias_shape.numel()

                weight = thetas[idx : idx + weight_size].reshape(weight_shape)
                idx += weight_size
                bias = thetas[idx : idx + bias_size].reshape(bias_shape)
                idx += bias_size

                # Assign parameters
                module.weight.data = weight
                module.bias.data = bias

        # Dummy input: a column vector of ones with shape (1, n_features)
        x = torch.ones(1, self.model[0].in_features)
        out = self.model(x)
        return out

    def run(self, thetas: Iterable[float]) -> float:
        """
        Convenience wrapper that returns the mean scalar prediction.
        """
        with torch.no_grad():
            out = self.forward(thetas)
        return float(out.mean().item())


__all__ = ["FullyConnectedLayer"]
