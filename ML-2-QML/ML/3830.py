"""Hybrid estimator that leverages a 128‑dimensional classical backbone and optional quantum embeddings."""
from __future__ import annotations

import torch
from torch import nn

class EstimatorQNN128(nn.Module):
    """
    Classical neural network that optionally consumes quantum expectation values.

    Parameters
    ----------
    input_dim : int, default 128
        Dimensionality of the raw feature vector.
    use_quantum : bool, default True
        If True, the network expects a second input containing the
        two quantum expectation values produced by the QML partner.
    """

    def __init__(self, input_dim: int = 128, use_quantum: bool = True) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        hidden = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Linear(2 if use_quantum else 32, 1)
        self.feature_extractor = hidden

    def forward(self, x: torch.Tensor, q_out: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw input of shape (batch, 128).
        q_out : torch.Tensor, optional
            Quantum expectation values of shape (batch, 2). Required if
            ``use_quantum`` is True.

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        h = self.feature_extractor(x)
        if self.use_quantum:
            if q_out is None:
                raise ValueError("q_out must be provided when use_quantum=True")
            out = self.head(q_out)
        else:
            out = self.head(h)
        return out

def EstimatorQNN() -> EstimatorQNN128:
    """Return a pre‑configured instance of EstimatorQNN128."""
    return EstimatorQNN128()

__all__ = ["EstimatorQNN128", "EstimatorQNN"]
