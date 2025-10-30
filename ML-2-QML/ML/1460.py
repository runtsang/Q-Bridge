"""Enhanced classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """
    A configurable feed‑forward network suitable for binary classification.
    Features optional batch‑norm, dropout, and residual‑style connectivity.
    """

    __all__ = ["QuantumClassifierModel"]

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        depth: int = 3,
        dropout: float = 0.0,
        batchnorm: bool = False,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the input.
        hidden_size : int, default=64
            Width of the hidden layers.
        depth : int, default=3
            Number of hidden layers.
        dropout : float, default=0.0
            Dropout probability (0.0 disables dropout).
        batchnorm : bool, default=False
            Whether to insert BatchNorm1d after each hidden layer.
        device : str, default="cpu"
            Target device for the network.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.device = torch.device(device)

        layers: List[nn.Module] = []
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_size)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())

            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))

            in_dim = hidden_size

        # Output head
        head = nn.Linear(in_dim, 2)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers).to(self.device)
        self.observables = [0, 1]  # indices of output logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.network(x.to(self.device))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    # Auxiliary introspection methods

    def get_encoding(self) -> List[int]:
        """Indices of input features used by the network."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Number of parameters per trainable layer."""
        return self.weight_sizes

    def get_observables(self) -> List[int]:
        """Placeholder for output node indices."""
        return self.observables
