"""Classical LSTM with autoencoder preprocessing and hybrid classification head.

This module defines :class:`GenQLSTM`, a drop‑in replacement for the original
QLSTM that remains purely classical while borrowing architectural ideas from
the quantum seed.  The model first projects the input embeddings into a
lower‑dimensional latent space with a lightweight MLP autoencoder, then
processes the sequence with a standard :class:`torch.nn.LSTM`, and finally
produces class probabilities through a hybrid head that emulates the
quantum expectation layer using a differentiable sigmoid.

The design mirrors the quantum implementation but stays on the CPU/GPU,
making it suitable for large‑scale experiments that want to compare
quantum‑enhanced and classical baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class HybridHead(nn.Module):
    """Differentiable sigmoid head that acts as a stand‑in for a quantum
    expectation value.  It can be replaced by a true quantum layer
    without changing the surrounding code."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return probs


@dataclass
class GenQLSTMConfig:
    """Configuration for the hybrid GenQLSTM model."""
    embedding_dim: int
    hidden_dim: int
    vocab_size: int
    tagset_size: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    shift: float = 0.0


class GenQLSTM(nn.Module):
    """Hybrid classical LSTM with autoencoder preprocessing and hybrid head."""
    def __init__(self, config: GenQLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.autoencoder = AutoencoderNet(
            config.embedding_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        self.lstm = nn.LSTM(
            config.latent_dim,
            config.hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.hidden_dim, config.tagset_size)
        self.hybrid_head = HybridHead(config.tagset_size, shift=config.shift)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: (seq_len, batch)
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        seq_len, batch, embed = embeds.shape
        flat = embeds.reshape(seq_len * batch, embed)
        reduced = self.autoencoder.encode(flat)  # (seq_len*batch, latent)
        reduced = reduced.reshape(seq_len, batch, -1)  # (seq_len, batch, latent)
        lstm_out, _ = self.lstm(reduced)  # (seq_len, batch, hidden)
        logits = self.classifier(lstm_out)  # (seq_len, batch, tagset)
        probs = self.hybrid_head(logits)  # (seq_len, batch, tagset)
        # return log‑softmax for compatibility
        return F.log_softmax(probs, dim=-1)


__all__ = ["GenQLSTM", "GenQLSTMConfig", "AutoencoderNet", "HybridHead"]
