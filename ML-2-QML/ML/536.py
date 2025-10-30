"""Extended classical LSTM with dropout, residual, and optional hybrid gates.

This module implements a drop‑in replacement for the original QLSTM but adds several classical
enhancements that make it more suitable for large‑scale sequence tagging experiments.

Key features
------------
* Dropout applied to the hidden state before each gate to mitigate overfitting.
* Residual connection between the input and hidden state at each time step.
* LayerNorm on the concatenated input to improve training stability.
* Hybrid gate option: each gate can optionally be a small MLP instead of a single linear layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QLSTM", "LSTMTagger"]

class QLSTM(nn.Module):
    """Classical LSTM cell with dropout, residual, and optional hybrid gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        hybrid: bool = False,
        gate_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual = nn.Identity()  # placeholder for future residual scaling
        self.ln = nn.LayerNorm(input_dim + hidden_dim)

        # Gate definitions
        self._gate_factory = self._hybrid_gate if hybrid else self._linear_gate
        self.forget = self._gate_factory(gate_hidden)
        self.input = self._gate_factory(gate_hidden)
        self.update = self._gate_factory(gate_hidden)
        self.output = self._gate_factory(gate_hidden)

    def _linear_gate(self, hidden: int | None = None) -> nn.Linear:
        return nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

    def _hybrid_gate(self, hidden: int | None = None) -> nn.Sequential:
        hidden = hidden or self.hidden_dim // 2
        return nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.hidden_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            combined = self.ln(combined)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.residual(hx) + x  # residual connection
            hx = self.dropout(hx)
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
    """Sequence tagging model that uses the enhanced QLSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        dropout: float = 0.0,
        hybrid_gate: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            dropout=dropout,
            hybrid=hybrid_gate,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
