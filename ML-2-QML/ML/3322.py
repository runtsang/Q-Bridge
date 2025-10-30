"""Hybrid LSTM tagger with optional classical convolutional encoder.

This module defines a pure‑Python implementation of QLSTMHybrid that
mirrors the quantum‑enhanced architecture but remains entirely classical.
It supports an optional 1‑D convolutional encoder that can be toggled
via the ``use_conv`` flag. The design is compatible with the
original QLSTM.py interface and can be used as a direct drop‑in
replacement when a quantum backend is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTMHybrid(nn.Module):
    """Classical hybrid LSTM tagger.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Vocabulary size for the embedding layer.
    tagset_size : int
        Number of target tags.
    use_conv : bool, optional
        If True, prepend a 1‑D convolutional encoder to the embeddings.
    conv_out_channels : int, optional
        Number of output channels for the convolutional encoder.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_conv: bool = False,
        conv_out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.use_conv = use_conv
        if use_conv:
            # 1‑D conv over the sequence dimension
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=conv_out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            )
            lstm_input_dim = conv_out_channels
        else:
            self.conv_encoder = None
            lstm_input_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            batch_first=False,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len, batch) containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag scores of shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)

        if self.use_conv:
            # conv expects (batch, channels, seq_len)
            conv_input = embeds.permute(1, 2, 0)  # (batch, embedding_dim, seq_len)
            conv_out = self.conv_encoder(conv_input)  # (batch, conv_out_channels, seq_len//2)
            # reshape back to (seq_len//2, batch, conv_out_channels)
            conv_out = conv_out.permute(2, 0, 1)
            lstm_out, _ = self.lstm(conv_out)
            tag_logits = self.hidden2tag(lstm_out)
            return F.log_softmax(tag_logits, dim=2)
        else:
            lstm_out, _ = self.lstm(embeds)
            tag_logits = self.hidden2tag(lstm_out)
            return F.log_softmax(tag_logits, dim=2)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, "
            f"use_conv={self.use_conv})"
        )


__all__ = ["QLSTMHybrid"]
