"""Hybrid classical CNN+LSTM tagging model inspired by QLSTM and QuantumNAT.

The model extracts 4‑dimensional features from each image using a small
convolutional network and then processes the resulting sequence with a
classical LSTM.  The public API matches that of the original QLSTM
implementation, enabling drop‑in replacement in existing pipelines.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Classical CNN feature extractor, adapted from the QFCModel
class ClassicalCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(x.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out)


# Classical LSTM tagger (identical to the one in the anchor QLSTM)
class LSTMTagger(nn.Module):
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# Hybrid pipeline: CNN → sequence → LSTMTagger
class QLSTMGenML(nn.Module):
    """Classical pipeline that chains a CNN feature extractor with an LSTMTagger."""

    def __init__(
        self,
        hidden_dim: int,
        tagset_size: int,
        vocab_size: int = 1000,
    ) -> None:
        super().__init__()
        self.cnn = ClassicalCNN()
        self.lstm_tagger = LSTMTagger(
            embedding_dim=4,  # CNN outputs 4‑dimensional vectors
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            tagset_size=tagset_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape (seq_len, batch, 1, 28, 28) representing a sequence
            of grayscale images.

        Returns
        -------
        torch.Tensor
            Log‑probabilities over tags for each time step.
        """
        seq_len, batch, c, h, w = images.shape
        # process each image through the CNN
        cnn_out = []
        for t in range(seq_len):
            img = images[t]
            feat = self.cnn(img)  # (batch, 4)
            cnn_out.append(feat.unsqueeze(0))
        cnn_seq = torch.cat(cnn_out, dim=0)  # (seq_len, batch, 4)
        # use the LSTMTagger on the CNN features
        logits = self.lstm_tagger(cnn_seq)
        return logits


__all__ = ["QLSTMGenML"]
