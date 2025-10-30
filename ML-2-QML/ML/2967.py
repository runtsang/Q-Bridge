"""Hybrid LSTM tagger with classical convolution and LSTM.

This module provides a drop‑in replacement for the original QLSTM tagger
while keeping everything classical. It exposes the same public API and
allows optional use of a lightweight 2‑D convolutional front‑end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq, embed = x.shape
        side = int(embed ** 0.5)
        x = x.view(batch * seq, 1, side, side)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3]).view(batch, seq, -1)

class HybridQLSTMTagger(nn.Module):
    """Classical hybrid tagger with convolution and LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv = Conv2DFilter(kernel_size, conv_threshold)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence shape: (batch, seq_len)
        embeds = self.word_embeddings(sentence)  # (batch, seq_len, embed_dim)
        conv_out = self.conv(embeds)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(conv_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["HybridQLSTMTagger"]
