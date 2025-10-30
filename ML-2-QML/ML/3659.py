"""Hybrid model: classical quanvolution filter followed by classical LSTM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """Convolutional filter that downsamples 28×28 grayscale images to 4×14×14 feature map."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.conv(x)          # [batch, out_channels, 14, 14]
        return self.flatten(features)    # [batch, out_channels*14*14]

class QuanvolutionQLSTM(nn.Module):
    """Classical quanvolution + LSTM hybrid for sequence classification."""
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10,
                 lstm_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        seq_len = 14 * 14
        feature_dim = 4
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        flat = self.filter(x)                    # [batch, seq_len*feature_dim]
        seq = flat.view(batch_size, 14 * 14, 4)   # [batch, seq_len, feature_dim]
        _, (hn, _) = self.lstm(seq)              # hn: [num_layers, batch, hidden_dim]
        logits = self.classifier(hn[-1])          # take last layer hidden state
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionQLSTM"]
