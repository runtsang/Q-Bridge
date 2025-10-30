"""Hybrid kernel–LSTM regression model with classical and quantum interfaces.

This module implements :class:`HybridKernelLSTMRegressor` that can be
interchanged with the quantum variant defined in the sibling module.
The classical version builds on the RBF kernel, a linear‑gate LSTM and
a small feed‑forward regressor.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

# Import the classical RBF kernel and the linear‑gate LSTM
from.QuantumKernelMethod import Kernel
from.QLSTM import QLSTM
from.EstimatorQNN import EstimatorQNN


class HybridKernelLSTMRegressor(nn.Module):
    """Classical hybrid kernel‑LSTM regressor.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector.
    hidden_dim : int
        Hidden size of the LSTM.
    output_dim : int
        Dimensionality of the regression output.
    n_qubits : int, optional
        Ignored in the classical variant but kept for API compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        # Classical RBF kernel
        self.kernel = Kernel(gamma=1.0)
        # Linear‑gate LSTM mirrors the classical QLSTM implementation
        self.lstm = QLSTM(input_dim=1, hidden_dim=hidden_dim, n_qubits=0)
        # Feed‑forward head
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``(seq_len, batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch, output_dim)``.
        """
        seq_len, batch, _ = x.shape
        # Compute a scalar RBF feature per timestep
        features = torch.stack(
            [self.kernel(x[i], x[i]).squeeze(-1) for i in range(seq_len)], dim=0
        ).unsqueeze(-1)  # shape (seq_len, batch, 1)
        lstm_out, _ = self.lstm(features)
        # Use final hidden state as prediction signal
        return self.head(lstm_out[-1])


__all__ = ["HybridKernelLSTMRegressor"]
