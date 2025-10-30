"""Enhanced classical fully‑connected layer with trainable parameters and dropout."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class FullyConnectedLayer(nn.Module):
    """
    A flexible multi‑layer perceptron that accepts a flattened parameter vector
    via :meth:`run`.  The network consists of an input layer, a hidden ReLU
    block with dropout, and a single output neuron.  This design mirrors the
    classical side of a quantum‑classical hybrid while remaining fully
    differentiable.
    """

    def __init__(self, n_features: int = 1, hidden_units: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )
        # Store the total number of trainable parameters for validation
        self._param_size = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def set_parameters_from_vector(self, theta: Iterable[float]) -> None:
        """
        Load a flattened parameter vector into the network.  The vector must
        match the size of all trainable parameters.
        """
        theta_tensor = torch.as_tensor(list(theta), dtype=torch.float32)
        if theta_tensor.numel()!= self._param_size:
            raise ValueError(
                f"Expected {self._param_size} parameters, got {theta_tensor.numel()}"
            )
        vector_to_parameters(theta_tensor, self.parameters())

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the network on a zero‑input tensor after loading the provided
        parameter vector.  Returns the mean output as a NumPy array to
        emulate the quantum ``run`` API.
        """
        self.set_parameters_from_vector(thetas)
        # Dummy input: zeros with the correct feature dimension
        x = torch.zeros((1, next(p.size() for p in self.parameters() if p.shape[1] == self.net[0].in_features)))
        out = self.forward(x)
        expectation = out.mean().detach().numpy()
        return np.array([expectation])


__all__ = ["FullyConnectedLayer"]
