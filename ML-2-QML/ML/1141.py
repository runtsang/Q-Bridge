"""Enhanced classical fully‑connected layer with dropout and batch‑norm.

The original seed provided a single linear layer with a tanh activation.
This extension introduces a configurable two‑layer feed‑forward network
with dropout and batch‑norm, allowing richer representation learning
while keeping the API (`run`) identical for compatibility.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class FCL(nn.Module):
    """
    Two‑layer fully‑connected network with optional dropout and batch‑norm.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dim : int, default 16
        Size of the hidden layer.
    dropout : float, default 0.0
        Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Run the network on a vector of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Sequence of parameters that will be reshaped to match the
            weight matrix of the first linear layer. The remaining
            parameters are ignored (they can be used for future
            extensions).

        Returns
        -------
        torch.Tensor
            The network output as a 1‑D tensor.
        """
        # Convert to tensor and reshape to match first layer weights
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
        expected_shape = self.linear1.weight.shape
        if theta_tensor.numel()!= expected_shape.numel():
            raise ValueError(
                f"Expected {expected_shape.numel()} parameters for the first "
                f"layer, got {theta_tensor.numel()}"
            )
        # Broadcast parameters to weight and bias
        weight = theta_tensor.view_as(self.linear1.weight)
        bias = torch.zeros_like(self.linear1.bias)
        self.linear1.weight.data = weight
        self.linear1.bias.data = bias

        # Forward pass
        x = torch.zeros((1, self.linear1.in_features))
        output = self.forward(x)
        return output.detach()


__all__ = ["FCL"]
