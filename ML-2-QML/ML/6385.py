"""Hybrid classical estimator that feeds a low‑dimensional embedding into a quantum circuit."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["EstimatorQNN"]


class EstimatorQNN(nn.Module):
    """
    A two‑stage model: a tiny neural network maps inputs to a 2‑D embedding,
    which is then used as the input parameter of a variational quantum circuit.
    The quantum circuit is defined in the companion quantum module and
    executed on the backend of your choice.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        embedding_dim: int = 2,
        qnum: int = 1,
        backend: str = "qiskit.aer.noise.NoiseModel",
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.qnum = qnum
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Embedding of shape (..., embedding_dim) that can be fed into the
            quantum estimator.
        """
        return self.embed(x)
