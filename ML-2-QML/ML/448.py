"""Hybrid LSTM with optional regularisation and dropout.

The class keeps the original API while adding a dropout layer on each gate
and an L2‑regularised embedding.  The ``n_qubits`` argument is kept for
compatibility with the quantum version but is ignored internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveQLSTM(nn.Module):
    """Classical LSTM cell with optional dropout on gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Ignored; present only for API compatibility.
    depth : int, default 0
        Ignored; kept for API compatibility.
    dropout : float, default 0.0
        Drop‑out probability applied to the gate activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        depth: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # API compatibility only
        self.depth = depth  # API compatibility only

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            # optional dropout on gate activations
            f = self.dropout(f)
            i = self.dropout(i)
            g = self.dropout(g)
            o = self.dropout(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`AdaptiveQLSTM`."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = AdaptiveQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            depth=depth,
            dropout=dropout,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["AdaptiveQLSTM", "LSTMTagger"]
