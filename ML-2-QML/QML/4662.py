"""Quantum‑enhanced hybrid kernel‑LSTM regressor.

This variant replaces the classical RBF kernel, the linear‑gate LSTM and
the feed‑forward head with their quantum counterparts.
The model remains drop‑in compatible with the classical implementation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchquantum as tq
# Import the quantum kernel and quantum LSTM
from.QuantumKernelMethod import Kernel
from.QLSTM import QLSTM
# EstimatorQNN is optional but kept for API completeness
from.EstimatorQNN import EstimatorQNN


class HybridKernelLSTMRegressor(tq.QuantumModule):
    """Quantum hybrid kernel‑LSTM regressor.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector.
    hidden_dim : int
        Hidden size of the LSTM.
    output_dim : int
        Dimensionality of the regression output.
    n_qubits : int
        Number of qubits used by the quantum circuits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        # Quantum RBF‑style kernel (TorchQuantum implementation)
        self.kernel = Kernel()
        # Quantum LSTM with small quantum circuits per gate
        self.lstm = QLSTM(input_dim=1, hidden_dim=hidden_dim, n_qubits=n_qubits)
        # Classical linear head to produce the final regression value
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
        # Quantum kernel produces a scalar per timestep
        features = torch.stack(
            [self.kernel(x[i], x[i]).squeeze(-1) for i in range(seq_len)], dim=0
        ).unsqueeze(-1)  # shape (seq_len, batch, 1)
        lstm_out, _ = self.lstm(features)
        return self.head(lstm_out[-1])


__all__ = ["HybridKernelLSTMRegressor"]
