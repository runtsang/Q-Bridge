"""Enhanced classical LSTM with dropout and weight sharing for robust training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """Drop‑in replacement for a classical LSTM cell with optional dropout and weight sharing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        dropout: float | None = None,
        shared_weights: bool = False,
    ) -> None:
        """Create an LSTM cell with optional dropout on the gates and a flag to share
        weights across the forget, input, and output gates.  The ``n_qubits`` argument
        remains present for API compatibility but is ignored in the classical variant."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.shared_weights = shared_weights

        gate_dim = hidden_dim
        # If weights are shared, we create a single linear layer for all gates
        if self.shared_weights:
            self.shared_linear = nn.Linear(input_dim + hidden_dim, gate_dim * 4)
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.shared_weights:
                gates = self.shared_linear(combined)
                f, i, g, o = gates.chunk(4, dim=1)
            else:
                f = self.forget_linear(combined)
                i = self.input_linear(combined)
                g = self.update_linear(combined)
                o = self.output_linear(combined)

            if self.dropout:
                f, i, g, o = (self.dropout(t) for t in (f, i, g, o))

            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

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
        dropout: float | None = None,
        shared_weights: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # Classical fallback for compatibility
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
                shared_weights=shared_weights,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
