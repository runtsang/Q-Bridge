"""Hybrid classical LSTM module with autoencoding preprocessing.

The class :class:`HybridQLSTM` is a drop‑in replacement for the original
QLSTM.  It keeps the same constructor signature but accepts an optional
autoencoder that compresses the input embeddings before they enter the
LSTM gates.  The class preserves the public API of the original module
so that existing scripts continue to run unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight fully‑connected autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(input_dim: int, **kwargs) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = AutoencoderConfig(input_dim=input_dim, **kwargs)
    return AutoencoderNet(cfg)


class HybridQLSTM(nn.Module):
    """Classical LSTM cell that optionally compresses inputs via an autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 autoencoder: Optional[AutoencoderNet] = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoencoder = autoencoder
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            if self.autoencoder is not None:
                x = self.autoencoder(x)
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use HybridQLSTM or nn.LSTM."""

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 autoencoder: Optional[AutoencoderNet] = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                                    autoencoder=autoencoder)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger", "Autoencoder", "AutoencoderNet", "AutoencoderConfig"]
