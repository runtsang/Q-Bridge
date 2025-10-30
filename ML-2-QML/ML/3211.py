"""Hybrid QCNN-QLSTM model – classical implementation."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, List

__all__ = ["UnifiedQCNNQLSTM", "UnifiedQCNNQLSTMConfig"]

class UnifiedQCNNQLSTMConfig:
    """Configuration for the hybrid QCNN-QLSTM model."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        qcnn_layers: int = 3,
        lstm_layers: int = 1,
        dropout: float = 0.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.qcnn_layers = qcnn_layers
        self.lstm_layers = lstm_layers
        self.dropout = dropout

class UnifiedQCNNQLSTM(nn.Module):
    """Classical hybrid QCNN‑QLSTM.

    The model first transforms the input vector with a dense feature map,
    then applies a stack of QCNN‑style convolution‑pooling blocks.  The
    resulting representation is fed into a multi‑layer LSTM, and the final
    hidden state is projected to a scalar output.

    Parameters
    ----------
    config : UnifiedQCNNQLSTMConfig or dict
        Configuration object or dictionary.
    """

    def __init__(self, config: UnifiedQCNNQLSTMConfig | dict | None = None):
        super().__init__()
        if config is None:
            config = UnifiedQCNNQLSTMConfig()
        if isinstance(config, dict):
            config = UnifiedQCNNQLSTMConfig(**config)
        self.config = config

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.Tanh(),
        )

        # QCNN blocks
        self.qcnn_blocks = nn.ModuleList()
        for _ in range(self.config.qcnn_layers):
            self.qcnn_blocks.append(self._qcnn_block())

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
        )

        # Output head
        self.head = nn.Linear(self.config.hidden_dim, 1)

    def _qcnn_block(self) -> nn.Module:
        """A single QCNN‑style convolution‑pooling block implemented with
        classical dense layers.  The block emulates the behaviour of the
        quantum circuit by alternating convolution and pooling operations.
        """
        conv = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        pool = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.Tanh(),
        )
        return nn.Sequential(conv, pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1).
        """
        batch, seq_len, _ = x.shape
        # Process each time step through QCNN blocks
        features = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h = self.feature_map(xt)
            for block in self.qcnn_blocks:
                h = block(h)
            features.append(h.unsqueeze(1))
        # Shape: (batch, seq_len, hidden_dim)
        seq_features = torch.cat(features, dim=1)
        # LSTM
        lstm_out, _ = self.lstm(seq_features)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        out = self.head(last_hidden)
        return torch.sigmoid(out)
