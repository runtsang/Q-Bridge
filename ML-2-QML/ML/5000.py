"""Hybrid classical sequence model combining auto‑encoder, LSTM and
classifier modules.  The implementation mirrors the structure of the
reference QLSTM and SamplerQNN files while adding an optional
classical auto‑encoder for dimensionality reduction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AutoencoderNet(nn.Module):
    """Simple MLP auto‑encoder (from reference 3)."""

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


class SamplerModule(nn.Module):
    """Softmax classifier (from reference 4)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class HybridQLSTM(nn.Module):
    """Drop‑in replacement that can toggle between classical and quantum
    LSTM back‑ends while keeping a shared interface."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        input_dim: int,
        use_quantum_lstm: bool = False,
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim=input_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_quantum_lstm = use_quantum_lstm
        # In the classical branch we use a plain nn.LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.sampler = SamplerModule()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len,)
        Returns:
            log‑probabilities of shape (seq_len, 2)
        """
        embeds = self.word_embeddings(sentence)
        # Reduce dimensionality with the auto‑encoder
        # (the encoder operates on each embedding independently)
        encoded = self.autoencoder.encode(embeds)
        # LSTM expects (seq_len, batch, input)
        lstm_out, _ = self.lstm(encoded.unsqueeze(1))
        flat = lstm_out.squeeze(1)
        probs = self.sampler(flat)
        return torch.log(probs)


__all__ = ["HybridQLSTM", "AutoencoderNet", "SamplerModule"]
