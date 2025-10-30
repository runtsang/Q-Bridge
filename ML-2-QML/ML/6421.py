"""
Classical implementation of HybridQLSTM featuring a QCNN‑style feature extractor followed by a standard LSTM.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """Fully‑connected surrogate for a quantum convolutional net.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the input embeddings.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridQLSTM(nn.Module):
    """Drop‑in replacement for the original LSTMTagger that augments embeddings with a QCNN‑style encoder."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.feature_extractor = QCNNModel(embedding_dim)
        # The LSTM now expects a 1‑dimensional feature per token
        self.lstm = nn.LSTM(1, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # ``sentence`` shape: (seq_len, batch)
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        seq_len, batch, _ = embeds.size()
        # Flatten for feature extraction
        flat = embeds.view(seq_len * batch, -1)
        features = self.feature_extractor(flat)  # (seq_len*batch, 1)
        # Reshape back to sequence format
        features = features.view(seq_len, batch, 1)
        lstm_out, _ = self.lstm(features)
        tag_logits = self.hidden2tag(lstm_out.view(seq_len, -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "QCNNModel"]
