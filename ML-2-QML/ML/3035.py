"""Hybrid classical LSTM with optional auto‑encoder preprocessing.

The module extends the original `QLSTM` implementation by adding an
optional auto‑encoder that reduces the dimensionality of the input
embeddings before they are fed to the LSTM.  This mirrors the
quantum‑auto‑encoder idea from the QML seed, while keeping the
entire pipeline fully classical.

Classes
-------
QLSTM : nn.Module
    Classical LSTM cell that can optionally prepend an Autoencoder.
LSTMTagger : nn.Module
    Sequence tagging model that uses QLSTM and an optional Autoencoder.

Functions
---------
Autoencoder : factory that returns an AutoencoderNet.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical auto‑encoder implementation.
# It is assumed to live in the same package as this module.
from.Autoencoder import Autoencoder, AutoencoderNet


class QLSTM(nn.Module):
    """Drop‑in classical LSTM cell that may prepend an Autoencoder.

    Parameters
    ----------
    input_dim : int
        Size of the input vectors.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int, optional
        Ignored in the classical implementation but kept for API
        compatibility with the quantum variant.
    autoencoder : Optional[AutoencoderNet]
        If provided, the encoder part of this auto‑encoder is applied
        to all inputs before the LSTM gates are computed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        autoencoder: Optional[AutoencoderNet] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.autoencoder = autoencoder

        # If an auto‑encoder is supplied, use its latent dimension
        # as the effective input size for the gates.
        effective_input_dim = (
            autoencoder.latent_dim if autoencoder is not None else input_dim
        )

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(effective_input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(effective_input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(effective_input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(effective_input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM over a sequence.

        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape ``(seq_len, batch, input_dim)``.
        states : Tuple[torch.Tensor, torch.Tensor] or None
            Optional initial hidden and cell states.

        Returns
        -------
        outputs : torch.Tensor
            LSTM outputs of shape ``(seq_len, batch, hidden_dim)``.
        final_states : Tuple[torch.Tensor, torch.Tensor]
            Final hidden and cell states.
        """
        if self.autoencoder is not None:
            # Apply the encoder to all time‑steps.
            inputs = self.autoencoder.encode(inputs)

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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`QLSTM` with optional auto‑encoder.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Size of the LSTM hidden state.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of target tags.
    n_qubits : int, optional
        When >0 the quantum LSTM branch is selected.
    autoencoder : Optional[AutoencoderNet]
        Auto‑encoder used for feature extraction.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        autoencoder: Optional[AutoencoderNet] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            # The quantum branch is not available in this file – the
            # corresponding implementation resides in the QML module.
            raise NotImplementedError(
                "Quantum LSTM branch requires the QML module."
            )
        else:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                autoencoder=autoencoder,
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices of shape ``(seq_len, batch)``.

        Returns
        -------
        torch.Tensor
            Log‑softmax over tag predictions of shape
            ``(seq_len, batch, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
