"""Hybrid classical regressor with optional quantum post‑processing.

The class implements a shallow MLP that mirrors the original EstimatorQNN
architecture but adds a configurable dropout and ReLU activations for
better regularisation.  It is intentionally lightweight so that it can be
used as a drop‑in replacement in classical pipelines or as the
feature‑extractor for the quantum module in the companion QML file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class HybridEstimatorQNN(nn.Module):
    """
    Classical regression network with a two‑layer MLP.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input.
    hidden_dim : int, default 8
        Size of the hidden layer.
    drop_rate : float, default 0.0
        Dropout probability applied after the hidden layer.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted scalar of shape (batch, 1).
        """
        return self.backbone(x)

__all__ = ["HybridEstimatorQNN"]
