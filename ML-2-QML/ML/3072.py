"""HybridEstimatorFCL: Classical implementation inspired by FCL and EstimatorQNN.

The class integrates a simple feed‑forward network (from EstimatorQNN) with a
parameterised linear layer (from FCL) to provide a classical baseline that can
be used as a surrogate for the quantum layer.  The `run` method accepts a list
of parameters and returns the network’s scalar output, mimicking the
quantum‑layer API.  The `forward` method exposes the network as a standard
PyTorch module.

This design allows a seamless switch between classical and quantum
implementations while keeping the public interface identical.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable


class HybridEstimatorFCL(nn.Module):
    """
    Classical hybrid estimator.

    Parameters
    ----------
    n_features : int, default 2
        Number of input features expected by the underlying network.
    n_qubits : int, default 1
        Number of qubits used in the quantum counterpart (ignored here but
        retained for API compatibility).
    """

    def __init__(self, n_features: int = 2, n_qubits: int = 1) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits

        # Feed‑forward network inspired by EstimatorQNN
        self.net = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

        # Auxiliary linear layer to mimic the simple FCL behaviour
        self.linear = nn.Linear(n_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        return self.net(inputs)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the quantum layer API.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters interpreted as input features.  If the length differs
            from ``n_features`` the vector is padded or truncated.

        Returns
        -------
        np.ndarray
            A 1‑D array containing the scalar expectation value.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32)

        # Align dimensionality
        if values.ndim == 1:
            values = values.view(1, -1)
        if values.shape[1]!= self.n_features:
            diff = self.n_features - values.shape[1]
            if diff > 0:
                pad = torch.zeros(values.shape[0], diff, dtype=values.dtype)
                values = torch.cat([values, pad], dim=1)
            else:
                values = values[:, : self.n_features]

        # Compute the expectation via the auxiliary linear layer
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


def FCL() -> HybridEstimatorFCL:
    """Return an instance of the hybrid estimator for backward compatibility."""
    return HybridEstimatorFCL()
