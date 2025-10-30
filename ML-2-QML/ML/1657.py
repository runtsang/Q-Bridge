"""Enhanced classical LSTM with multi‑layer support and dropout."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class QLSTM(nn.Module):
    """
    Classical LSTM that matches the public API of the original
    quantum version.  It can be used as a drop‑in replacement
    and adds multi‑layer and dropout support.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Size of each input vector.
        hidden_dim : int
            Size of the hidden state.
        n_qubits : int, optional
            Ignored – kept for API compatibility.  If non‑zero
            the class behaves exactly like the classical one.
        num_layers : int, default 1
            Number of stacked LSTM layers.
        dropout : float, default 0.0
            Dropout probability applied between layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # We deliberately ignore ``n_qubits`` – the classical implementation
        # does not use quantum gates.
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states : tuple or None
            Initial hidden and cell states.  If ``None`` zero
            states are used.  The format is the same as for
            ``torch.nn.LSTM``.
        """
        return self.lstm(inputs, states)


__all__ = ["QLSTM"]
