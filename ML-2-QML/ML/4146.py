"""
Hybrid classical regressor with optional residual MLP and LSTM gating.

This module defines EstimatorQNNFusion as a pure PyTorch nn.Module.
It can be used as a drop‑in replacement for the original EstimatorQNN
while exposing deeper residual structures and sequence handling.

The implementation deliberately avoids any quantum imports so that it
remains lightweight and can run on CPU/GPU.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List

class ResidualBlock(nn.Module):
    """Two‑layer residual unit with ReLU."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.fc2(y)
        return self.act(x + y)

class ResidualMLP(nn.Module):
    """Stack of ResidualBlock units."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class EstimatorQNNFusion(nn.Module):
    """
    Hybrid regressor that combines a deep residual MLP with optional
    LSTM gating and a quantum feature map (provided by the QML module).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64, 32],
        use_lstm: bool = False,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.residual_mlp = ResidualMLP(input_dim, hidden_dims)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dims[-1],
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            self.lstm_out = nn.Linear(lstm_hidden, hidden_dims[-1])
        else:
            self.lstm = None

        self.final = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, features) or (batch, seq_len, features)
            if ``use_lstm`` is True.
        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        if self.use_lstm:
            # Expecting (batch, seq_len, features)
            seq_out, _ = self.lstm(x)
            seq_out = seq_out[:, -1, :]
            seq_out = self.lstm_out(seq_out)
            mlp_out = self.residual_mlp(seq_out)
        else:
            mlp_out = self.residual_mlp(x)
        return self.final(mlp_out)

__all__ = ["EstimatorQNNFusion"]
