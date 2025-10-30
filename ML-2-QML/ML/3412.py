"""Classical implementation of a hybrid QLSTM + Quanvolution image classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Simple 2×2 convolution that mimics a Quanvolution layer.
    Produces a sequence of 4‑dimensional vectors (one per patch).
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.conv(x)  # (batch, 4, 14, 14)
        # Reshape to (batch, seq_len=196, 4)
        return features.view(x.size(0), 4, -1).transpose(1, 2)


class QLSTMQuanvolutionClassifier(nn.Module):
    """
    Classical hybrid model that first extracts 2×2 patches via a
    simple convolution (QuanvolutionFilter) and then processes the
    resulting sequence of 4‑dimensional vectors with a gated LSTM.
    """
    def __init__(self, hidden_dim: int = 128, n_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.lstm = nn.LSTM(input_size=4,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        seq = self.qfilter(x)          # (batch, seq_len=196, 4)
        lstm_out, _ = self.lstm(seq)   # (batch, seq_len, hidden_dim)
        logits = self.classifier(lstm_out[:, -1, :])  # last time step
        return F.log_softmax(logits, dim=-1)


__all__ = ["QLSTMQuanvolutionClassifier"]
