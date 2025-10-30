"""Hybrid classical model combining quanvolution filter and LSTM for sequence classification/tagging.

This implementation stays close to the original `Quanvolution` and `QLSTM` seeds, but uses purely classical operations so it can run on any CPU/GPU backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuanvolutionQLSTMHybrid"]

class ClassicalQuanvolutionFilter(nn.Module):
    """
    Classical 2×2 patch extractor with a trainable linear projection.
    """
    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=2, stride=2)
        self.proj = nn.Linear(out_channels, out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.conv(x)
        # flatten to (batch, 4 * 14 * 14)
        return patches.view(x.size(0), -1)

class ClassicalLSTM(nn.Module):
    """
    Classical LSTM cell that mimics the interface of the quantum LSTM
    but uses a standard `nn.LSTM`.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return out

class QuanvolutionQLSTMHybrid(nn.Module):
    """
    Hybrid model that first extracts features via the classical quanvolution
    filter and then feeds the per‑patch sequence into a classical LSTM.
    The model can be used for image classification or for tagging if the
    sequence dim is set appropriately.
    """
    def __init__(self,
                 num_classes: int = 10,
                 hidden_dim: int = 256,
                 vocab_size: int = 30522,
                 tagset_size: int = 10) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.lstm = ClassicalLSTM(self.filter.conv.out_channels, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.tag_head = nn.Linear(hidden_dim, tagset_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        feat = self.filter(x)                     # (B, 4*14*14)
        seq = feat.view(x.size(0), -1, self.filter.conv.out_channels)  # (B, 196, 4)
        out = self.lstm(seq)                      # (B, 196, hidden_dim)
        logits = self.class_head(out[:, -1, :])   # take last time step
        return F.log_softmax(logits, dim=-1)
