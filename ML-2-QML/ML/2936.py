"""Hybrid model that combines QCNN and LSTM in a classical architecture.

The model first applies a stack of fully connected layers that mimic the
hierarchical quantum convolution–pooling structure of the original QCNN.
The extracted features are then fed into a recurrent network (an
`nn.LSTM`) that learns temporal dependencies.  This design allows the
model to be used as a drop‑in replacement for either the QCNN or the
QLSTM pipelines, while providing richer spatial‑temporal feature
extraction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["QCNNQLSTMHybrid"]

class QCNNQLSTMHybrid(nn.Module):
    """
    Classical hybrid of QCNN feature extractor and LSTM temporal model.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input vector.
    hidden_dim : int
        Size of the hidden representation in the LSTM.
    n_qubits : int
        Number of qubits in the original QCNN.  It is kept only for
        compatibility with the quantum version; it does not influence
        the classical computation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        # QCNN‑style fully‑connected hierarchy
        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),          # feature_map
            nn.Linear(16, 16), nn.Tanh(),        # conv1
            nn.Linear(16, 12), nn.Tanh(),        # pool1
            nn.Linear(12, 8), nn.Tanh(),         # conv2
            nn.Linear(8, 4), nn.Tanh(),          # pool2
            nn.Linear(4, 4), nn.Tanh(),          # conv3
            nn.Linear(4, 1)                      # head
        )
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(input_dim=1, hidden_size=hidden_dim, batch_first=True)
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, 1).
        """
        batch, seq_len, _ = x.shape
        # Flatten time dimension to process each time step independently
        x_flat = x.reshape(batch * seq_len, -1)
        feats = self.feature_extractor(x_flat).reshape(batch, seq_len, -1)
        # Pass through LSTM
        lstm_out, _ = self.lstm(feats)
        # Classifier
        out = self.classifier(lstm_out)
        return torch.sigmoid(out)
