"""Hybrid image‑sequence model combining a classical quanvolution filter with a linear LSTM encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "QLSTM",
    "LSTMTagger",
    "QuanvolutionQLSTM",
]


class QuanvolutionFilter(nn.Module):
    """Classical 2‑channel conv that maps 2×2 patches to 4‑dim features."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            feat = self.conv(x)
            feat = feat.view(b * t, -1)
            return feat.view(b, t, -1)
        else:
            feat = self.conv(x)
            return feat.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the quanvolution filter followed by a linear head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.filter(x)
        logits = self.linear(feat)
        return F.log_softmax(logits, dim=-1)


class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates (drop‑in replacement for the quantum version)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(inputs, states)


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses either the classical QLSTM or nn.LSTM."""

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
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


class QuanvolutionQLSTM(nn.Module):
    """Hybrid model that processes a sequence of images with a classical quanvolution filter
    followed by a linear LSTM encoder and a classifier head.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    hidden_dim : int
        Hidden size of the LSTM.
    """

    def __init__(self, num_classes: int = 10, hidden_dim: int = 128) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.lstm = nn.LSTM(
            input_size=4 * 14 * 14,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        batch, seq_len, c, h, w = x.shape
        x_reshaped = x.view(batch * seq_len, c, h, w)
        feats = self.filter(x_reshaped)  # (batch*seq_len, 4*14*14)
        feats = feats.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(feats)
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return F.log_softmax(logits, dim=-1)
