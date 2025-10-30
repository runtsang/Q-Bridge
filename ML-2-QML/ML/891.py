"""
Classical fully‑connected layer with configurable hidden depth, dropout, and batch‑norm.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np


class FCLayer(nn.Module):
    """
    A flexible fully‑connected neural network layer.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : Sequence[int], optional
        Sizes of hidden layers. If omitted, a single linear layer is used.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to insert a BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Sequence[int] | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        in_features = n_features
        hidden_sizes = hidden_sizes or []

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = h

        # Final linear mapping to a single scalar output
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original API by accepting a sequence of parameters and
        returning the mean activation over a single‑sample batch.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameters that will be reshaped to match the
            network's linear layers. The sequence length must equal the
            total number of learnable weights plus biases.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the mean output.
        """
        # Flatten the network's parameters into a single vector
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        flat_params = torch.cat(params)

        # Ensure the input matches the expected size
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
        if theta_tensor.shape[0]!= flat_params.shape[0]:
            raise ValueError(
                f"Expected {flat_params.shape[0]} parameters, got {theta_tensor.shape[0]}"
            )

        # Overwrite the network's parameters with the supplied thetas
        idx = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(theta_tensor[idx : idx + numel].view_as(param))
            idx += numel

        # Run a single‑sample forward pass
        x = torch.randn(1, self.model[0].in_features)
        out = self.forward(x)

        # Return the mean activation as a NumPy array
        return out.mean().detach().numpy()


__all__ = ["FCLayer"]
