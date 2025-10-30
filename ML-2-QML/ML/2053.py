"""Enhanced classical fully connected layer with optional dropout and batch norm.

This module defines a FullyConnectedLayer that can handle batched inputs,
supports configurable activation, dropout, and optional batch normalization.
The run method accepts a batch of inputs and returns the transformed output.
"""

import numpy as np
import torch
from torch import nn
from typing import Optional

class FullyConnectedLayer(nn.Module):
    """A flexible fully‑connected layer for regression or classification tasks."""
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        activation: Optional[str] = "tanh",
        use_dropout: bool = False,
        dropout_prob: float = 0.1,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = self._get_activation(activation)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_prob)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_features)

    def _get_activation(self, name: Optional[str]):
        if name is None:
            return None
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        if self.use_batchnorm:
            out = self.batchnorm(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Convert a 1‑D array of parameters into a batch of inputs,
        run the linear transformation and return the mean output
        (mimicking the quantum expectation value).
        """
        batch = torch.as_tensor(thetas, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            out = self.forward(batch)
        return out.mean(dim=0).detach().numpy()
