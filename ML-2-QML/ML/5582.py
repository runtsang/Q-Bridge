"""Hybrid quantum‑inspired convolutional auto‑encoder with optional quantum LSTM.

This module combines:
* QCNN‑style feature extraction (linear layers with tanh activations).
* A fully‑connected layer acting as a linear classifier.
* An auto‑encoder that compresses intermediate representations.
* Either a classical LSTM or a quantum‑enhanced LSTM (from :mod:`QLSTM`).

The design is compatible with the original QCNN helper but offers richer
hierarchies and sequence modelling.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# Import helper classes from the reference seed modules
from.Autoencoder import AutoencoderNet, AutoencoderConfig
from.QLSTM import QLSTM

class HybridQCNNModel(nn.Module):
    """Hybrid QCNN–Autoencoder–LSTM architecture."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        # QCNN‑style feature extractor
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Auto‑encoder for compression
        auto_config = AutoencoderConfig(
            input_dim=4,
            latent_dim=latent_dim,
            hidden_dims=(8, 4),
            dropout=0.1,
        )
        self.autoencoder = AutoencoderNet(auto_config)

        # Fully‑connected layer
        self.fcl = nn.Linear(latent_dim, 1)

        # Sequence model
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim=1, hidden_dim=lstm_hidden, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True
            )

        # Final output
        self.output = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # QCNN feature extraction
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Auto‑encoder compression
        z = self.autoencoder.encode(x)

        # Fully‑connected processing
        y = self.fcl(z)

        # Sequence modelling (treat batch as sequence of length 1)
        y = y.unsqueeze(1)  # (batch, seq_len=1, feature=1)
        lstm_out, _ = self.lstm(y)
        out = self.output(lstm_out.squeeze(1))
        return torch.sigmoid(out)

def HybridQCNN(
    input_dim: int = 8,
    *,
    hidden_dim: int = 16,
    latent_dim: int = 8,
    lstm_hidden: int = 32,
    lstm_layers: int = 1,
    n_qubits: int = 0,
) -> HybridQCNNModel:
    """Convenience factory mirroring the quantum helper."""
    return HybridQCNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        n_qubits=n_qubits,
    )

__all__ = ["HybridQCNN", "HybridQCNNModel"]
