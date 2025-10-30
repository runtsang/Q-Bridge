"""Enhanced classical LSTM with optional dropout.

This module keeps the original ``QLSTM`` API while adding a ``dropout`` argument
to the constructor.  Dropout is applied to the hidden state after each
time step, allowing for regularisation without changing the rest of the
pipeline.  The surrounding ``LSTMTagger`` now exposes a ``dropout`` parameter
that is forwarded to the underlying LSTM implementation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """
    Classical LSTM cell mirroring the quantum interface but with
    dropout support.  The constructor signature matches the seed
    implementation, but an additional ``dropout`` argument is
    accepted.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Linear projections for the four LSTM gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise hidden and cell states to zeros.

        Returns a tuple ``(hx, cx)`` each of shape
        ``(batch, hidden_dim)``.
        """
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h0, c0

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : Tensor
            ``(seq_len, batch, input_dim)``
        states : Tuple[Tensor, Tensor] | None
            Optional external hidden and cell states.

        Returns
        -------
        outputs : Tensor
            ``(seq_len, batch, hidden_dim)``
        (h_n, c_n) : Tuple[Tensor, Tensor]
            Final hidden and cell states.
        """
        seq_len, batch_size, _ = inputs.shape
        device = inputs.device

        if states is None:
            hx, cx = self._init_states(batch_size, device)
        else:
            hx, cx = states

        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]  # (batch, input_dim)
            combined = torch.cat([x_t, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Apply dropout to the hidden state
            hx = self.dropout(hx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use either the classical ``QLSTM`` or
    PyTorch's built‑in ``nn.LSTM``.  A ``dropout`` argument is exposed and
    passed to the underlying LSTM implementation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                dropout=dropout if dropout > 0.0 else 0.0,
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.

        Parameters
        ----------
        sentence : Tensor
            ``(seq_len, batch)`` word indices.

        Returns
        -------
        Tensor
            Log‑softmaxed tag scores of shape ``(seq_len, batch, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM", "LSTMTagger"]
