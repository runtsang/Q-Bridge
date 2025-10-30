"""Hybrid classical module combining a convolutional backbone, a classical quanvolution filter, and a classical LSTM head.

Author: OpenAI GPT‑OSS‑20B
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQuanvolutionQLSTM"]

class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2‑channel convolution followed by a 2×2 patch extractor.

    The filter mimics the behaviour of the original `QuanvolutionFilter`
    but is implemented purely with PyTorch layers for maximum speed.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        out = self.conv(x)          # (B, out_channels, 14, 14)
        out = self.bn(out)
        out = self.dropout(out)
        # reshape to a sequence of patches: (B, seq_len=196, feature_dim=out_channels)
        B, C, H, W = out.shape
        seq_len = H * W
        return out.permute(0, 2, 3, 1).reshape(B, seq_len, C)

class ClassicalQLSTM(nn.Module):
    """A lightweight LSTM wrapper that uses a single nn.LSTM layer.

    The class is kept separate to mirror the structure of the quantum
    implementation in the QML module, enabling easy swapping of the
    underlying recurrent cell.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, seq_len, input_dim)
        _, (hn, _) = self.lstm(seq)
        # hn: (1, B, hidden_dim) -> squeeze to (B, hidden_dim)
        return hn.squeeze(0)

class HybridQuanvolutionQLSTM(nn.Module):
    """Full classical pipeline: quanvolution filter → LSTM → classifier.

    Parameters
    ----------
    hidden_dim : int
        Size of the hidden state in the LSTM.
    num_classes : int
        Number of target classes (default 10 for MNIST).
    """
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.lstm = ClassicalQLSTM(input_dim=4, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, num_classes).
        """
        seq = self.filter(x)
        hidden = self.lstm(seq)
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=-1)
