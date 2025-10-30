"""Hybrid classical estimator that simulates quantum layer behavior.

The module defines a PyTorch neural network that mirrors the
structure of the original EstimatorQNN but replaces the quantum
circuit with a classical surrogate.  The network consists of
an encoder, a simulated quantum layer, and a regression head,
providing a single prediction for each input pair.
"""

from __future__ import annotations

import torch
from torch import nn


class ClassicalQuantumLayer(nn.Module):
    """Simulated quantum layer used in the classical estimator."""

    def __init__(self, n_features: int, n_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a quantum‑like expectation value.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, n_outputs) after a tanh activation.
        """
        return torch.tanh(self.linear(x))


class HybridEstimator(nn.Module):
    """
    Hybrid estimator combining a classical encoder, a simulated quantum
    layer, and a regression head.

    Architecture:
        2 → 8 (encoder) → 8 → 4 (quantum layer) → 4 → 1 (decoder)
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
        )
        self.quantum_layer = ClassicalQuantumLayer(n_features=8, n_outputs=4)
        self.decoder = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid estimator.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) with the regression prediction.
        """
        encoded = self.encoder(inputs)
        quantum_out = self.quantum_layer(encoded)
        return self.decoder(quantum_out)


__all__ = ["HybridEstimator"]
