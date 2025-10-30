"""Hybrid classical architecture combining a lightweight 2‑D convolution with a standard LSTM for sequence classification.

The module defines:
- QuanvolutionFilter: a lightweight 2‑D convolution that maps 1‑channel images to 4‑channel feature maps.
- QuanvolutionClassifier: a simple classifier that uses the filter followed by a linear head.
- QuanvolutionQLSTM: a drop‑in replacement that accepts a sequence of images, applies the filter to each image, and feeds the resulting feature vectors into a standard LSTM before projecting to tag logits.

Typical usage::

    >>> from quanvolution_gen101 import QuanvolutionQLSTM
    >>> model = QuanvolutionQLSTM(input_dim=1, hidden_dim=64, vocab_size=10000, tagset_size=10)
    >>> logits = model(x)  # x shape: (batch, seq_len, 1, 28, 28)
    >>> loss = F.cross_entropy(logits, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QuanvolutionFilter(nn.Module):
    """A lightweight 2‑D convolution that maps 1‑channel images to 4‑channel feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28) -> (batch, 4, 14, 14)
        return self.conv(x).view(x.size(0), -1)  # flatten to (batch, 4*14*14)

class QuanvolutionClassifier(nn.Module):
    """Simple classifier that uses the quanvolution filter followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class QuanvolutionQLSTM(nn.Module):
    """
    Hybrid sequence classifier that combines a quanvolution filter with a standard LSTM.
    The input is expected to be a batch of sequences of images with shape
    (batch, seq_len, 1, 28, 28).  The filter is applied to each image in the
    sequence, producing a feature vector that is fed into an LSTM.  Finally a
    linear layer projects the LSTM outputs to tag logits.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector (default 4*14*14).
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary (unused in the classical version but kept for API compatibility).
    tagset_size : int
        Number of output tags.
    conv_quantum : bool, optional
        Flag to enable a quantum quanvolution filter (ignored in the classical version).
    lstm_quantum : bool, optional
        Flag to enable a quantum LSTM (ignored in the classical version).
    """
    def __init__(
        self,
        input_dim: int = 4 * 14 * 14,
        hidden_dim: int = 64,
        vocab_size: int = 10000,
        tagset_size: int = 10,
        conv_quantum: bool = False,
        lstm_quantum: bool = False,
    ) -> None:
        super().__init__()
        # The classical filter is always used in this implementation.
        self.qfilter = QuanvolutionFilter()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (seq_len, batch, tagset_size).
        """
        batch, seq_len, c, h, w = x.shape
        # Flatten the sequence dimension into the batch dimension
        x_flat = x.view(batch * seq_len, c, h, w)
        features = self.qfilter(x_flat)  # (batch*seq_len, feature_dim)
        # Reshape back to a sequence for the LSTM
        features_seq = features.view(batch, seq_len, -1).transpose(0, 1)  # (seq_len, batch, feature_dim)
        lstm_out, _ = self.lstm(features_seq)
        logits = self.hidden2tag(lstm_out)  # (seq_len, batch, tagset_size)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuanvolutionQLSTM"]
