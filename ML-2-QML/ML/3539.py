"""QuanvolutionQLSTMHybrid – classical implementation.

This module defines a hybrid neural network that combines a
classical 2×2 convolutional filter, a classical LSTM over the
resulting patches, and a linear classifier head.  It mirrors the
original `Quanvolution` example but uses only PyTorch primitives
and can be dropped into any standard training pipeline.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuanvolutionFilter", "QLSTM", "QuanvolutionQLSTMHybrid"]


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that downsamples a 28×28 image to 14×14 patches."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Input shape: (batch, 1, 28, 28) → (batch, 4, 14, 14)
        return self.conv(x)


class QLSTM(nn.Module):
    """Pure PyTorch LSTM wrapper – identical to nn.LSTM but keeps the name."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return out  # (batch, seq_len, hidden_dim)


class QuanvolutionQLSTMHybrid(nn.Module):
    """Hybrid model that applies a classical filter, an LSTM over patches,
    and a linear classifier."""

    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        # LSTM over 196 patches of size 4
        self.lstm = QLSTM(input_dim=4, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract patches
        features = self.filter(x)  # (batch, 4, 14, 14)
        batch, channels, h, w = features.shape
        # Reshape to (batch, seq_len=196, input_dim=4)
        seq = features.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        # Pass through LSTM
        lstm_out = self.lstm(seq)  # (batch, 196, hidden_dim)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        logits = self.classifier(last_hidden)
        return F.log_softmax(logits, dim=-1)
