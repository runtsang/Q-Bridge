"""Hybrid classical LSTM with convolutional feature extraction for sequence tagging."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """Classical LSTM that first extracts a 4‑dimensional feature vector from each image
    using a CNN (inspired by Quantum‑NAT) and then feeds these vectors into a
    standard nn.LSTM.  The structure mirrors the quantum variant but keeps all
    operations on the CPU/GPU, making it a drop‑in replacement for the quantum
    implementation."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        # The CNN feature extractor produces four values per image
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 4),
            nn.BatchNorm1d(4),
        )
        # The LSTM consumes the 4‑dimensional feature vectors
        self.lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        self.norm = nn.BatchNorm1d(hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.shape[1]
        device = inputs.device
        hx = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        cx = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # inputs shape: (seq_len, batch, 1, 28, 28)
        seq_len, batch, _, _, _ = inputs.shape
        imgs = inputs.view(seq_len * batch, 1, 28, 28)
        feats = self.features(imgs)          # (seq_len*batch, 4)
        feats_seq = feats.view(seq_len, batch, 4)
        hx, cx = self._init_states(feats_seq, states)
        lstm_out, (hn, cn) = self.lstm(feats_seq, (hx, cx))
        return self.norm(lstm_out), (hn, cn)


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid QLSTM for feature extraction
    and sequence modelling."""
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
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (batch, seq_len, embedding_dim)
        # QLSTM expects (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(embeds.permute(1, 0, 2))
        lstm_out = lstm_out.permute(1, 0, 2)  # back to (batch, seq_len, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QLSTM", "LSTMTagger"]
