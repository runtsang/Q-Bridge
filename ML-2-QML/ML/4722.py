"""Hybrid classical LSTM model with autoencoder and optional quantum gates.

This module defines a `HybridQLSTM` class that combines a classical
embedding layer, a fully‑connected autoencoder for dimensionality
reduction, and either a pure PyTorch LSTM or a quantum‑enhanced LSTM
(`QLSTM`).  The design follows the structure of the original
`QLSTM.py` while incorporating the autoencoder and convolution
filters from the other reference pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Autoencoder utilities (from Autoencoder.py)
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
    """Standard MLP autoencoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int,
                *, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    """Convenience factory mirroring the original helper."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Classical LSTM implementation (from QLSTM.py)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in classical LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
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

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Tagger that switches between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Hybrid model combining all components
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """A multi‑stage LSTM that embeds, autoencodes, and then processes
    sequences with a classical or quantum LSTM cell.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    latent_dim : int, default 32
        Latent dimension of the autoencoder.
    n_qubits : int, default 0
        If >0, the LSTM gates are realised by a quantum circuit.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 latent_dim: int = 32, n_qubits: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Autoencoder that compresses each embedding to `latent_dim`
        self.autoencoder = Autoencoder(embedding_dim,
                                       latent_dim=latent_dim)
        # Choose between classical and quantum LSTM
        if n_qubits > 0:
            self.lstm = QLSTM(latent_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(latent_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        sentence : torch.LongTensor
            Tensor of token indices with shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for each tag.
        """
        embeds = self.word_embeddings(sentence)          # (seq_len, batch, embed_dim)
        seq_len, batch, _ = embeds.shape
        # Flatten to feed the autoencoder
        flat = embeds.view(seq_len * batch, -1)
        latents = self.autoencoder.encode(flat).view(seq_len, batch, -1)  # (seq_len, batch, latent_dim)
        lstm_out, _ = self.lstm(latents)
        tag_logits = self.hidden2tag(lstm_out.view(seq_len * batch, -1))
        return F.log_softmax(tag_logits, dim=1).view(seq_len, batch, -1)

__all__ = ["HybridQLSTM", "Autoencoder", "AutoencoderConfig",
           "AutoencoderNet", "QLSTM", "LSTMTagger"]
