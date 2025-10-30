"""Hybrid classical LSTM with auto‑encoder preprocessing and sampler‑based stochastic gating.

The module mirrors the original `QLSTM.py` API but augments the
classical LSTM cell with two new components:

* :class:`AutoencoderNet` – compresses the input embeddings before
  they reach the gates.
* :class:`SamplerQNN` – generates a 4‑dimensional mask that
  stochastically modulates the forget, input, update and output gates.

Both components are fully differentiable and can be trained jointly
with the LSTM.  The :class:`LSTMTagger` keeps the same interface as
the original tagger, allowing a seamless switch between a vanilla
``nn.LSTM`` and this hybrid version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Auto‑encoder utilities
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """A lightweight fully‑connected auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# SamplerQNN – a lightweight stochastic mask generator
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """Generates a 4‑dimensional stochastic mask."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # The input ``x`` is ignored; the network learns a fixed mask
        # that can be tuned during training.
        return F.softmax(self.net(torch.ones(1, 4, device=x.device)), dim=-1)


# --------------------------------------------------------------------------- #
# Classical hybrid LSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in classical LSTM that optionally injects stochastic masks
    from a sampler and compresses inputs via an auto‑encoder."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, latent_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        # Auto‑encoder for feature compression
        latent = latent_dim or hidden_dim
        self.autoencoder = AutoencoderNet(AutoencoderConfig(input_dim,
                                                            latent_dim=latent))
        # Sampler for stochastic gating
        self.sampler = SamplerQNN()

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            # Compress input via auto‑encoder
            comp = self.autoencoder.encode(x)
            combined = torch.cat([comp, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            # Optional stochastic mask from sampler
            mask = self.sampler(combined)  # shape (1,4)
            f *= mask[0, 0]
            i *= mask[0, 1]
            g *= mask[0, 2]
            o *= mask[0, 3]

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


# --------------------------------------------------------------------------- #
# Tagger that can switch between the hybrid LSTM and a vanilla nn.LSTM
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the hybrid LSTM
    and a vanilla ``nn.LSTM``."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 latent_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits, latent_dim=latent_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "AutoencoderNet", "AutoencoderConfig", "SamplerQNN"]
