"""Hybrid LSTM with QCNN-inspired feature extractor.

This module implements a purely classical sequence tagging network that
combines a conventional LSTM with a lightweight QCNN-style
fully‑connected stack.  The QCNN block acts as a feature extractor
before the recurrent layer, mirroring the quantum variant while
remaining fully differentiable with PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """Classical convolution‑inspired feature extractor.

    Parameters
    ----------
    input_dim : int, default 8
        Dimensionality of the input word embeddings.
    output_dim : int, default 8
        Dimensionality of the output features fed to the LSTM.
    """
    def __init__(self, input_dim: int = 8, output_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, output_dim), nn.Tanh())
        self.head = nn.Linear(output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


class HybridQLSTMQCNN(nn.Module):
    """Sequence‑tagging model that fuses an LSTM with a QCNN feature extractor.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Dimensionality of the LSTM hidden state.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.conv = QCNNModel(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)          # shape: [seq_len, batch, embed]
        features = self.conv(embeds)                    # shape: [seq_len, batch, hidden]
        lstm_out, _ = self.lstm(features)               # shape: [seq_len, batch, hidden]
        tag_logits = self.hidden2tag(lstm_out)           # shape: [seq_len, batch, tagset]
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QCNNModel", "HybridQLSTMQCNN"]
