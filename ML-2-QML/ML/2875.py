"""Hybrid classical LSTM model with a convolutional front‑end for sequence tagging.

This module implements :class:`HybridQLSTM`, a lightweight PyTorch model that
first extracts 2×2 patches from each image in a sequence using a single
convolutional layer, flattens the resulting feature map, and feeds the
sequence of feature vectors into a standard :class:`torch.nn.LSTM`.
A linear classifier maps the LSTM outputs to tag logits. The API
mirrors that of the original QLSTM/Quanvolution examples so it can be
used as a drop‑in replacement.

The class is intentionally minimal – it focuses on the architectural
combination of a convolutional filter and an LSTM, not on training
details.  All parameters are trainable via PyTorch’s autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQLSTM(nn.Module):
    """Classical hybrid model: Conv2d → LSTM → Linear classifier."""

    def __init__(self, hidden_dim: int, tagset_size: int) -> None:
        """
        Parameters
        ----------
        hidden_dim : int
            Hidden size of the LSTM.
        tagset_size : int
            Number of target tags (output classes).
        """
        super().__init__()
        # 2×2 convolution with stride 2 reduces 28×28 → 14×14
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Feature dimensionality after flattening
        self.feature_dim = 4 * 14 * 14
        # Classical LSTM processes the sequence of feature vectors
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        # Linear head that maps hidden state to tag logits
        self.classifier = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of image sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape ``(batch, seq_len, tagset_size)``.
        """
        batch, seq_len, c, h, w = x.shape
        # Collapse batch and sequence dimensions for the convolution
        x = x.view(batch * seq_len, c, h, w)
        feats = self.conv(x)                     # (batch*seq, 4, 14, 14)
        feats = feats.view(batch, seq_len, -1)   # (batch, seq, feature_dim)
        lstm_out, _ = self.lstm(feats)           # (batch, seq, hidden_dim)
        logits = self.classifier(lstm_out)       # (batch, seq, tagset_size)
        return F.log_softmax(logits, dim=-1)
