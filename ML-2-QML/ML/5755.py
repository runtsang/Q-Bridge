"""HybridEstimatorQNN: a classical model that emulates a quantum layer via trigonometric activations.

The architecture mirrors the original EstimatorQNN feed‑forward network and augments it with a
quantum‑inspired layer that uses sin/cos rotations to generate expectation‑like features.
This design allows the model to be trained with standard PyTorch optimisers while preserving
the expressive power of a variational quantum circuit.

The class is fully compatible with the original EstimatorQNN module – the ``EstimatorQNN`` function
in EstimatorQNN.py can be imported and used as a drop‑in replacement when a purely classical
implementation is required.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

class HybridEstimatorQNN(nn.Module):
    """
    A hybrid classical‑quantum‑inspired estimator.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input vector.
    hidden_dim : int, optional
        Size of the hidden layer in the classical encoder.
    """

    def __init__(self, n_features: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        # Quantum‑inspired layer parameters
        self.theta_in = nn.Parameter(torch.randn(hidden_dim // 2))
        self.theta_weight = nn.Parameter(torch.randn(hidden_dim // 2))
        # Read‑out linear head
        self.readout = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        The classical encoder produces a feature vector `h`.  The quantum‑inspired
        layer then applies sinusoidal rotations to each feature, mimicking the
        expectation value of a Pauli‑Y measurement on a parametrised qubit.

        Returns
        -------
        torch.Tensor
            Single‑output regression prediction.
        """
        h = self.encoder(x)
        # Quantum‑inspired rotation: sin(theta_in) * h + cos(theta_weight) * h
        q = torch.sin(self.theta_in) * h + torch.cos(self.theta_weight) * h
        out = self.readout(q)
        return out

    def run(self, inputs: Iterable[float]) -> float:
        """
        Convenience wrapper that accepts a plain Python iterable and returns a scalar.

        Parameters
        ----------
        inputs : Iterable[float]
            Input features.

        Returns
        -------
        float
            Prediction as a Python float.
        """
        with torch.no_grad():
            tensor = torch.tensor(list(inputs), dtype=torch.float32).unsqueeze(0)
            return float(self.forward(tensor).item())

__all__ = ["HybridEstimatorQNN"]
