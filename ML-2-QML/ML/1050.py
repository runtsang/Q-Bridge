"""Hybrid LSTM implementation with classical regularisation and optional dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class QLSTM(nn.Module):
    """Drop‑in replacement for the original QLSTM that keeps the same API but
    extends the classical LSTM with dropout and weight‑norm.  The module
    accepts a *dropout* flag and a *use_weight_norm* flag that, when set,
    add dropout to the gates and a weight‑norm wrapper around every linear
    layer.  The interface remains compatible with the seed."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = (
            weight_norm(nn.Linear(input_dim + hidden_dim, gate_dim))
            if use_weight_norm
            else nn.Linear(input_dim + hidden_dim, gate_dim)
        )
        self.input_linear = (
            weight_norm(nn.Linear(input_dim + hidden_dim, gate_dim))
            if use_weight_norm
            else nn.Linear(input_dim + hidden_dim, gate_dim)
        )
        self.update_linear = (
            weight_norm(nn.Linear(input_dim + hidden_dim, gate_dim))
            if use_weight_norm
            else nn.Linear(input_dim + hidden_dim, gate_dim)
        )
        self.output_linear = (
            weight_norm(nn.Linear(input_dim + hidden_dim, gate_dim))
            if use_weight_norm
            else nn.Linear(input_dim + hidden_dim, gate_dim)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional dropout and weight‑norm."""
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            if self.dropout is not None:
                f = self.dropout(f)
            i = torch.sigmoid(self.input_linear(combined))
            if self.dropout is not None:
                i = self.dropout(i)
            g = torch.tanh(self.update_linear(combined))
            if self.dropout is not None:
                g = self.dropout(g)
            o = torch.sigmoid(self.output_linear(combined))
            if self.dropout is not None:
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
    """Sequence tagging model that uses either :class:`QLSTM` or ``nn.LSTM``."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        use_weight_norm: bool = False,
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
                use_weight_norm=use_weight_norm,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
