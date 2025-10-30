"""
Enhanced classical estimator mirroring the original EstimatorQNN but with
additional regularisation and deeper architecture.  The model is fully
compatible with PyTorch training loops and can be plugged into any
standard scikit‑learn or PyTorch workflow.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class EnhancedEstimatorQNN(nn.Module):
    """
    Feed‑forward regressor with:
      * 3 hidden layers (input → 64 → 32 → 16 → output)
      * BatchNorm after each hidden layer
      * Dropout (0.2) to mitigate over‑fitting
      * Tanh activations (compatible with the original example)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int, int, int] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs: torch.Tensor
            Tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample.
        """
        return self.net(inputs)


def EstimatorQNN() -> EnhancedEstimatorQNN:
    """
    Factory returning an instance of the enhanced estimator.

    Returns
    -------
    EnhancedEstimatorQNN
        Initialized model ready for training.
    """
    return EnhancedEstimatorQNN()


__all__ = ["EnhancedEstimatorQNN", "EstimatorQNN"]
