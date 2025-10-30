"""Hybrid QCNN‑QLSTM architecture for classical experiments.

The module defines a drop‑in replacement for the original QCNN and
QLSTM seed projects.  The :class:`QCNNQLSTMHybrid` class first extracts
features using a fully‑connected QCNN analogue, then runs a classical
LSTM to model temporal dependencies, and finally produces tag logits.
The public API matches the seed modules so that the class can be
imported without modifications.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

# --------------------------------------------------------------------------- #
# Classical QCNN backbone
# --------------------------------------------------------------------------- #
class _QCNNBody(nn.Module):
    """Fully‑connected analogue of the QCNN layers.

    The architecture follows the same dimensional progression as the
    quantum QCNN: 8 → 16 → 12 → 8 → 4 → 4.  Tanh activations are used
    to preserve smooth gradients similar to expectation‑value outputs.
    """
    def __init__(self, in_features: int = 8, out_features: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --------------------------------------------------------------------------- #
# Classical LSTM wrapper
# --------------------------------------------------------------------------- #
class _ClassicLSTM(nn.Module):
    """Simple wrapper around :class:`torch.nn.LSTM`."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (seq_len, batch, input_dim)
        return self.lstm(x)

# --------------------------------------------------------------------------- #
# Hybrid QCNN‑QLSTM tagger
# --------------------------------------------------------------------------- #
class QCNNQLSTMHybrid(nn.Module):
    """Drop‑in replacement for the original LSTMTagger that uses a QCNN
    feature extractor followed by a classical LSTM.  The class accepts
    a ``n_qubits`` argument that, when set to a positive integer, can
    be extended to use a quantum LSTM in the quantum module.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.qcnn = _QCNNBody(in_features=embedding_dim, out_features=hidden_dim)
        # Use classical LSTM in the classical module
        self.lstm = _ClassicLSTM(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that runs the hybrid or classical LSTM.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of token indices with shape ``(seq_len, batch)``.
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        # Extract QCNN features per time step
        qcnn_features = torch.stack([self.qcnn(e) for e in embeds], dim=0)
        # qcnn_features shape: (seq_len, batch, hidden_dim)
        lstm_out, _ = self.lstm(qcnn_features)
        # Map to tag logits
        tag_logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(tag_logits, dim=-1)

__all__ = ["QCNNQLSTMHybrid"]
