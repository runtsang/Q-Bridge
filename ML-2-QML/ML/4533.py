"""Hybrid LSTM with attention and autoencoding – classical implementation.

The module mirrors the quantum interface but replaces all quantum gates with
classical linear layers and variational circuits.  It can be used as a
drop‑in replacement for the original `QLSTM` and `LSTMTagger` while
adding a self‑attention block and a fully‑connected autoencoder.  The
regression head operates on the latent representation produced by the
autoencoder.

Typical usage::

    model = HybridQLSTM(embedding_dim=50, hidden_dim=128,
                        vocab_size=10000, tagset_size=17,
                        latent_dim=32, n_qubits=0)
    tags, reg = model(torch.randint(0, 10000, (20, 1)))

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable, Optional, List


# --------------------------------------------------------------------------- #
#  Classical self‑attention – learnable parameters
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Self‑attention block with learnable rotation and entanglement matrices.

    The block receives a sequence of hidden states and produces an attention‑weighted
    representation.  The rotation and entanglement matrices are trainable
    parameters that control the query/key projection.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape: (seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (seq_len, embed_dim)
        """
        query = torch.matmul(inputs, self.rotation_params)
        key = torch.matmul(inputs, self.entangle_params)
        scores = torch.softmax(query @ key.t() / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)), dim=-1)
        return torch.matmul(scores, inputs)


# --------------------------------------------------------------------------- #
#  Classical autoencoder – fully‑connected
# --------------------------------------------------------------------------- #
class Autoencoder(nn.Module):
    """Simple MLP autoencoder used after the attention block."""

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
#  Classical LSTM cell (drop‑in replacement)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Linear‑gate LSTM that mimics the interface of the quantum LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
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
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
#  Hybrid model – classical + attention + autoencoder + regression
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Drop‑in replacement for the original `QLSTM` that adds attention, an autoencoder
    and a regression head.  The model can be instantiated in a purely classical
    mode (``n_qubits=0``) or with a quantum LSTM (``n_qubits>0``).
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 latent_dim: int = 32,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.attention = ClassicalSelfAttention(embedding_dim)
        self.autoencoder = Autoencoder(embedding_dim, latent_dim=latent_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Token indices of shape (seq_len, batch)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            * Tag logits (log‑softmax) of shape (seq_len, tagset_size)
            * Regression scalar of shape (1,)
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))  # (seq_len, batch, hidden)

        # Attention over the LSTM outputs – collapse batch dimension for simplicity
        attn_output = self.attention(lstm_out.squeeze(1))  # (seq_len, embed)

        # Autoencoder latent representation
        latent = self.autoencoder.encode(attn_output)  # (seq_len, latent)

        # Regression: mean over sequence
        reg_output = self.regressor(latent.mean(dim=0))  # (1,)

        # Tagging head
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))  # (seq_len, tagset)

        return F.log_softmax(tag_logits, dim=1), reg_output.squeeze(0)

__all__ = ["HybridQLSTM", "ClassicalQLSTM", "ClassicalSelfAttention", "Autoencoder"]
