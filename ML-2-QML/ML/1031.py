"""Enhanced classical LSTM with quantum‑style gating and regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

class QLSTM(nn.Module):
    """Drop‑in replacement that mimics quantum gates with a learnable bias and dropout.

    The gate equations are identical to a standard LSTM, but the linear
    projections are followed by a configurable dropout and a *classical*
    bias that emulates a quantum phase shift.  This allows a fair comparison
    with the original quantum version while keeping the model fully
    differentiable and compatible with any PyTorch optimiser.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 dropout: float = 0.0, gate_scale: float = 1.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = Dropout(dropout)
        self.gate_scale = gate_scale

        # Linear projections for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Classical bias that mimics a quantum phase shift
        self.phase_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Linear projections + dropout + bias
            f = torch.sigmoid(self.forget_linear(combined) * self.gate_scale + self.phase_bias)
            i = torch.sigmoid(self.input_linear(self.dropout(combined)) * self.gate_scale + self.phase_bias)
            g = torch.tanh(self.update_linear(combined) * self.gate_scale + self.phase_bias)
            o = torch.sigmoid(self.output_linear(combined) * self.gate_scale + self.phase_bias)

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
    """Sequence tagging using the enhanced QLSTM or a vanilla LSTM."""

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
        self.lstm = (
            QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                  dropout=dropout)
            if n_qubits > 0
            else nn.LSTM(embedding_dim, hidden_dim)
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return F.log_softmax(self.hidden2tag(lstm_out.view(len(sentence), -1)), dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
