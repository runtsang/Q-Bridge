"""Classical hybrid model combining a 2D convolutional filter and an LSTM for sequence classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter with stride 2 to extract local patches."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        features = self.conv(x)   # (batch, out_channels, H', W')
        # Flatten spatial dims to create a sequence
        batch, out_ch, h, w = features.shape
        seq_len = h * w
        return features.view(batch, seq_len, out_ch)

class ClassicalQLSTM(nn.Module):
    """Standard LSTM cell with optional dropout."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    def forward(self, seq: torch.Tensor, seq_lengths: torch.Tensor | None = None):
        if seq_lengths is not None:
            packed = pack_padded_sequence(seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(seq)
        return out

class QuanvolutionLSTMTagger(nn.Module):
    """
    Hybrid model that first applies a classical quanvolution filter to extract
    local patches, then feeds the resulting sequence into an LSTM for
    sequence tagging/classification.
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        self.lstm = ClassicalQLSTM(input_dim=4, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, height, width)
            seq_lengths: Optional lengths of each sequence after flattening patches.
        Returns:
            log probabilities over classes for each batch item.
        """
        seq = self.qfilter(x)  # shape: (batch, seq_len, 4)
        lstm_out = self.lstm(seq, seq_lengths)  # shape: (batch, seq_len, hidden_dim)
        # Use last hidden state for classification
        if seq_lengths is not None:
            idx = (seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, lstm_out.size(-1))
            last_hidden = lstm_out.gather(1, idx).squeeze(1)
        else:
            last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "ClassicalQLSTM", "QuanvolutionLSTMTagger"]
